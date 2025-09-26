#!/usr/bin/env python3


from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# -----------------------
# Env / sys path setup
# -----------------------
os.environ.setdefault("TORCH_HOME", "model_cache")
os.environ.setdefault("HF_HOME", "model_cache")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYGLET_HEADLESS", "1")
os.environ.setdefault("PYGLET_BACKEND", "egl")

sys.path.append("./evoworld")
sys.path.append("third_party/vggt/")

# -----------------------
# Project imports
# -----------------------
from evoworld.inference.navigator_evoworld import Navigator
from evoworld.inference.forward_evoworld import process_batch
from evoworld.reprojection.pano_to_pers_utils import calculate_segment_indices
from evoworld.trainer.unet_plucker import UNetSpatioTemporalConditionModel
from evoworld.pipeline.pipeline_evoworld import StableVideoDiffusionPipeline

from evoworld.reprojection.reproject_vggt_open3d_utils import (
    PointCloudProcessor,
    SceneBuilder,
    CubemapRenderer,
    predictions_to_target_view,
)

from third_party.vggt.vggt.models.vggt import VGGT
from third_party.vggt.vggt.utils.geometry import unproject_depth_map_to_point_map
from third_party.vggt.vggt.utils.load_fn import load_and_preprocess_images
from third_party.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri

from equilib import Equi2Pers

from utils.geometry import xyz_euler_to_four_by_four_matrix_batch
from utils.conversion import numpy_to_image
from utils.plucker_embedding import equirectangular_to_ray

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("UnifiedRefactor")

# -----------------------
# Constants
# -----------------------
UNITY_TO_OPENCV = np.array([1, -1, 1, -1, 1, -1], dtype=float)
DEFAULT_PANO_H, DEFAULT_PANO_W = 576, 1024
DEFAULT_PERS_H, DEFAULT_PERS_W = 384, 512

# -----------------------
# Small utilities
# -----------------------

def set_random_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    if not isinstance(x, torch.Tensor):
        return x
    # assume CHW in [-1,1]
    x = (x * 0.5 + 0.5).clamp(0, 1)
    x = (x.mul(255).byte().detach().cpu().numpy().transpose(1, 2, 0))
    return Image.fromarray(x)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    if not isinstance(img, Image.Image):
        return img
    arr = np.asarray(img)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t * 2 - 1


def save_frames(frames: List[torch.Tensor], out_dir: str, start_idx: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for i, fr in enumerate(frames):
        p = os.path.join(out_dir, f"{i + start_idx + 1:03}.png")
        tensor_to_pil(fr).save(p)


# -----------------------
# VGGT wrapper
# -----------------------
class VGGTProcessor:
    """Lightweight VGGT processor wrapper.

    Fixes the device bug from the original implementation.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.model = VGGT().to(self.device).eval()
        # Load weights
        # Prefer a local cache if present; otherwise download once via hub.
        url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        with LOGGER.catch_warnings():
            state = torch.hub.load_state_dict_from_url(url, map_location=self.device)
        self.model.load_state_dict(state)

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # images: (1, T, C, H, W) or (B, T, C, H, W) depending on loader.
        # load_and_preprocess_images already ensures a batch dim of 1.
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        with torch.autocast(device_type=device_type, dtype=dtype):
            return self.model(images)


# -----------------------
# Main pipeline
# -----------------------
class UnifiedLoopConsistencyPipeline:
    """Unified pipeline for loop consistency evaluation with VGGT reconstruction."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Models (lazy init)
        self.navigator: Optional[Navigator] = None
        self.vggt: Optional[VGGTProcessor] = None
        self.equi2pers: Optional[Equi2Pers] = None

        # Utilities
        self.point_processor = PointCloudProcessor()
        self.scene_builder = SceneBuilder()
        self.cubemap_renderer = CubemapRenderer()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing unified pipeline...")

    # ---------- Model init ----------
    def initialize_models(self) -> None:
        self.logger.info("Loading Navigator (UNet + SVD pipeline)...")
        self.navigator = Navigator()
        self.navigator.get_pipeline(
            self.args.unet_path,
            self.args.svd_path,
            model_height=DEFAULT_PANO_H,
            progress_bar=False,
            num_frames=self.args.num_frames,
        )

        if self.args.single_segment:
            self.logger.info("Loading VGGT model...")
            self.vggt = VGGTProcessor(self.device)

        self.logger.info("Initializing Equi2Pers...")
        self.equi2pers = Equi2Pers(
            height=DEFAULT_PERS_H,
            width=DEFAULT_PERS_W,
            fov_x=90,
            mode="bilinear",
        )
        self.logger.info("All models loaded successfully!")

    def setup_model_and_pipeline(self, args: argparse.Namespace):
        """Setup UNet model and SVD pipeline for the single-segment fast path."""
        weight_dtype = torch.float32

        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            args.unet_path, subfolder="unet", low_cpu_mem_usage=True
        )
        unet.requires_grad_(False)

        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            args.unet_path, unet=unet, local_files_only=True, low_cpu_mem_usage=True
        )
        pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)

        rays = equirectangular_to_ray(target_H=DEFAULT_PANO_H // 8, target_W=DEFAULT_PANO_W // 8)
        rays = torch.tensor(rays, dtype=weight_dtype, device=self.device)

        return pipeline, rays, weight_dtype

    # ---------- Data init ----------
    def determine_data_config(self) -> Tuple[str, bool]:
        try:
            folder_contents = os.listdir(self.args.base_folder)
        except FileNotFoundError as e:
            raise ValueError(f"Base folder '{self.args.base_folder}' not found") from e
        is_single_video = "panorama" in folder_contents
        data_root = self.args.base_folder if is_single_video else os.path.join(self.args.base_folder, "test")
        return data_root, is_single_video

    def create_dataset_and_loader(self, data_root: str, is_single_video: bool):
        # Set default dims
        self.args.height = DEFAULT_PANO_H
        self.args.width = DEFAULT_PANO_W

        sampling_method = "reprojection" if self.args.single_segment else "empty_with_traj"
        load_complete_episode = not self.args.single_segment

        from dataset.CameraTrajDataset import CameraTrajDataset
        dataset = CameraTrajDataset(
            data_root,
            width=self.args.width,
            height=self.args.height,
            trajectory_file=None,
            memory_sampling_args={"sampling_method": sampling_method, "include_initial_frame": True},
            load_complete_episode=load_complete_episode,
            reprojection_name="rendered_panorama_vggt_open3d",
            is_single_video=is_single_video,
        )

        loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
        return dataset, loader

    # ---------- Core steps ----------
    @torch.inference_mode()
    def process_segment_no_memory(self, batch: Dict[str, Any], segment_id: int) -> List[torch.Tensor]:
        """Generate the first segment (no memory)."""
        assert self.navigator is not None

        current_path = batch["cam_traj"].to(self.device).squeeze(0)
        images = batch["pixel_values"]  # (B=1, T, C, H, W)

        start_idx = segment_id * (self.args.num_frames - 1)
        end_idx = start_idx + self.args.num_frames

        memorized_images = torch.zeros_like(batch["memorized_pixel_values"][:, start_idx:end_idx]).to(self.device)
        start_image = images[0, 0].to(self.device)

        navigate_fn = getattr(self.navigator, "navigate_curve_path" if self.args.curve_path else "navigate_path")
        generations = navigate_fn(
            current_path,
            start_image,
            width=DEFAULT_PANO_W,
            height=DEFAULT_PANO_H,
            num_inference_steps=25,
            memorized_images=memorized_images,
            infer_segment=True,
            segment_id=segment_id,
        )

        navigation = [img for move in generations for img in move]
        return navigation

    @torch.inference_mode()
    def process_segment_with_memory(
        self,
        batch: Dict[str, Any],
        segment_id: int,
        start_image: torch.Tensor,
        memorized_pixel_values: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        assert self.navigator is not None

        current_path = batch["cam_traj"].to(self.device).squeeze(0)
        first_pixel_values = batch["pixel_values"][0, 0]
        memorized_pixel_values = [first_pixel_values] + memorized_pixel_values
        memorized_images = torch.stack(memorized_pixel_values).unsqueeze(0).to(self.device)

        navigate_fn = getattr(self.navigator, "navigate_curve_path" if self.args.curve_path else "navigate_path")
        generations = navigate_fn(
            current_path,
            start_image,
            width=DEFAULT_PANO_W,
            height=DEFAULT_PANO_H,
            num_inference_steps=25,
            memorized_images=memorized_images,
            infer_segment=True,
            segment_id=segment_id,
        )
        navigation = [img for move in generations for img in move]
        return navigation

    def convert_pano_to_pers(
        self, panoramic_frames: List[torch.Tensor], camera_params: np.ndarray, segment_id: int
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        assert self.equi2pers is not None

        perspective_frames: List[np.ndarray] = []
        target_yaws: List[float] = []

        # Original code used (segment_id + 1) * 24 + 24; keep as-is.
        look_at_idx = (segment_id + 1) * 24 + 24

        for i, frame_tensor in enumerate(panoramic_frames):
            frame_np = np.array(tensor_to_pil(frame_tensor))  # RGB uint8

            current_idx = i + 1
            if current_idx <= len(camera_params):
                current_yaw_deg = camera_params[current_idx - 1][4]
                look_at_point = camera_params[min(look_at_idx, len(camera_params) - 1)]
                target_yaw_rad = math.atan2(
                    look_at_point[0] - camera_params[current_idx - 1][0],
                    look_at_point[2] - camera_params[current_idx - 1][2],
                )
                yaw_diff = math.radians(current_yaw_deg) - target_yaw_rad
            else:
                yaw_diff = 0.0

            equi_img = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            equi_img = np.transpose(equi_img, (2, 0, 1))  # CHW for Equi2Pers
            pers_img = self.equi2pers(equi=equi_img, rots={"pitch": 0, "roll": 0, "yaw": yaw_diff})
            pers_img = np.transpose(pers_img, (1, 2, 0))  # HWC
            perspective_frames.append(pers_img)
            target_yaws.append(yaw_diff / math.pi * 180.0)

        return perspective_frames, np.array(target_yaws, dtype=float)

    def run_vggt_inference(self, perspective_frames: List[np.ndarray]) -> Dict[str, Any]:
        assert self.vggt is not None

        with tempfile.TemporaryDirectory() as tmp:
            img_paths: List[str] = []
            for i, frame in enumerate(perspective_frames):
                # assume RGB uint8
                pil_image = Image.fromarray(frame.astype(np.uint8))
                p = os.path.join(tmp, f"temp_{i:03d}.png")
                pil_image.save(p)
                img_paths.append(p)

            images = load_and_preprocess_images(img_paths).to(self.device)
            preds = self.vggt(images)

        # Convert pose encodings to matrices
        extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
        preds["extrinsic"] = extrinsic
        preds["intrinsic"] = intrinsic

        # To numpy (squeeze batch dim)
        out: Dict[str, Any] = {}
        for k, v in preds.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.detach().cpu().numpy().squeeze(0)
            else:
                out[k] = v

        # World points from depth
        depth_map = out["depth"]
        world_points = unproject_depth_map_to_point_map(depth_map, out["extrinsic"], out["intrinsic"])
        out["world_points_from_depth"] = world_points
        return out

    def load_camera_poses(self, episode_path: str) -> np.ndarray:
        cam_file = os.path.join(episode_path, "camera_poses.txt")
        if not os.path.isfile(cam_file):
            raise FileNotFoundError(f"camera_poses.txt not found under {episode_path}")

        rows: List[List[float]] = []
        with open(cam_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if parts and ("Frame" in parts[0] or "frame" in parts[0]):
                    # header row
                    continue
                if len(parts) >= 7:
                    # skip frame_id, take next 6 values
                    vals = [float(x) for x in parts[1:7]]
                    rows.append(vals)
        if not rows:
            raise ValueError(f"No valid camera pose rows parsed from {cam_file}")

        cam = np.asarray(rows, dtype=float)
        # Unity->OpenCV convention flip
        cam *= UNITY_TO_OPENCV
        return cam

    # ---------- Episode orchestration ----------
    def process_episode(self, batch: Dict[str, Any], episode: str, dataset) -> None:
        self.logger.info(f"\nProcessing episode: {episode}")

        episode_save_dir = os.path.join(self.args.save_dir, episode)
        os.makedirs(episode_save_dir, exist_ok=True)

        episode_path = batch["episode_path"][0]
        camera_params = self.load_camera_poses(episode_path)

        all_generated_frames: List[torch.Tensor] = []
        latest_memories: List[np.ndarray] = []  # will be filled after VGGT/renderer

        for segment_id in range(self.args.num_segments):
            self.logger.info(f"Processing segment {segment_id}")
            start_idx, end_idx, _ = calculate_segment_indices(segment_id)

            if segment_id == 0:
                generated_frames = self.process_segment_no_memory(batch, segment_id)
            else:
                # Use last generated frame as start image
                current_frame_pil = tensor_to_pil(all_generated_frames[-1])
                current_frame_tensor = pil_to_tensor(current_frame_pil).to(self.device)

                # memorized_pixel_values: transform each memory (numpy image) -> tensor
                memorized_pixel_values = [dataset.transform(numpy_to_image(m)) for m in latest_memories]
                generated_frames = self.process_segment_with_memory(
                    batch, segment_id, current_frame_tensor, memorized_pixel_values
                )

            if all_generated_frames:
                generated_frames = generated_frames[1:]  # avoid duplicating first frame
            all_generated_frames.extend(generated_frames)

            # Save predicted frames (optional)
            if self.args.save_frames:
                frames_path = os.path.join(episode_save_dir, f"predictions_{segment_id}")
                start_idx_seg = segment_id * (self.args.num_frames - 1)
                save_frames(generated_frames, frames_path, start_idx_seg)

            # For all but last segment: do reprojection + VGGT to build memories for next seg
            if segment_id < self.args.num_segments - 1:
                self.logger.info(f"Converting segment {segment_id} to perspective...")
                perspective_frames, target_yaws = self.convert_pano_to_pers(
                    all_generated_frames, camera_params, segment_id
                )

                # Dump perspective frames for debugging/inspection
                dump_dir = os.path.join(episode_save_dir, f"perspective_look_at_center_{segment_id}")
                os.makedirs(dump_dir, exist_ok=True)
                for idx, frm in enumerate(perspective_frames):
                    bgr = cv2.cvtColor(frm.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(dump_dir, f"{idx + 1:03}.png"), bgr)

                # Update camera yaw with computed target_yaws
                temp_cam = camera_params.copy()
                if len(target_yaws) > 0:
                    s = max(0, end_idx - len(target_yaws))
                    temp_cam[s:end_idx, 4] = target_yaws[: end_idx - s]

                self.logger.info(f"Running VGGT inference on segment {segment_id}...")
                preds = self.run_vggt_inference(perspective_frames)

                self.logger.info(f"Generating target views for segment {segment_id}...")
                cam_poses_tensor = torch.tensor(temp_cam, dtype=torch.float32, device=self.device)
                cam_poses_4x4 = xyz_euler_to_four_by_four_matrix_batch(cam_poses_tensor, relative=True)

                outdir = os.path.join(episode_save_dir, f"rendered_panorama_vggt_open3d_{segment_id}")

                latest_memories = predictions_to_target_view(
                    preds,
                    cam_poses_4x4.detach().cpu().numpy(),
                    conf_thres=50.0,
                    filter_by_frames="all",
                    mask_black_bg=False,
                    mask_white_bg=False,
                    show_cam=True,
                    mask_sky=False,
                    target_dir=episode_save_dir,
                    image_subdir=f"perspective_look_at_center_{segment_id}",
                    prediction_mode="depth_unproject",
                    num_target_view=24,
                    outdir=outdir,
                    only_render_last_24_frame=False,
                )

                # Free memory
                del preds, perspective_frames
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.logger.info(f"Episode {episode} processing completed!")

    @torch.inference_mode()
    def run_pipeline(self) -> None:
        set_random_seeds(self.args.seed)
        self.logger.info("Starting unified loop consistency pipeline...")
        self.logger.info(f"UNet Path: {self.args.unet_path}")
        self.logger.info(f"SVD Path:  {self.args.svd_path}")

        self.initialize_models()
        data_root, is_single_video = self.determine_data_config()
        val_dataset, val_loader = self.create_dataset_and_loader(data_root, is_single_video)

        for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            if idx < self.args.start_idx:
                continue
            if idx >= self.args.num_data + self.args.start_idx:
                break
            current_episode = val_dataset.episodes[idx]
            self.process_episode(batch, current_episode, val_dataset)

    @torch.inference_mode()
    def run_single_segment(self) -> None:
        set_random_seeds(self.args.seed)
        self.logger.info("Starting single-segment path...")
        self.logger.info(f"UNet Path: {self.args.unet_path}")
        self.logger.info(f"SVD Path:  {self.args.svd_path}")

        pipeline, rays, weight_dtype = self.setup_model_and_pipeline(self.args)
        data_root, is_single_video = self.determine_data_config()
        val_dataset, val_loader = self.create_dataset_and_loader(data_root, is_single_video)

        for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            if idx < self.args.start_idx:
                continue
            if idx >= self.args.num_data + self.args.start_idx:
                break
            current_episode = val_dataset.episodes[idx]
            episode_save_dir = os.path.join(self.args.save_dir, current_episode)
            os.makedirs(episode_save_dir, exist_ok=True)
            # ensure arg for process_batch
            self.args.mask_mem = False
            self.logger.info("")
            process_batch(batch, self.args, pipeline, rays, weight_dtype, episode_save_dir, current_episode)


# -----------------------
# CLI
# -----------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Loop Consistency Pipeline (Refactored)")

    # Model paths
    parser.add_argument("--unet_path", type=str, required=True, help="Path to UNet model")
    parser.add_argument(
        "--svd_path",
        type=str,
        default="stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        help="Path to SVD model",
    )

    # Data configuration
    parser.add_argument("--base_folder", type=str, default="data/Curve_Loop/test", help="Base folder containing episodes")
    parser.add_argument("--save_dir", type=str, default="unified_output", help="Output directory")
    parser.add_argument("--dataset_name", type=str, default="CameraTrajDataset", help="Dataset name")

    # Processing parameters
    parser.add_argument("--num_data", type=int, default=1, help="Number of episodes to process")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index")
    parser.add_argument("--num_segments", type=int, default=3, help="Number of segments to process")
    parser.add_argument("--num_frames", type=int, default=25, help="Frames per segment")
    parser.add_argument("--save_frames", action="store_true", help="Save intermediate frames")

    # Options
    parser.add_argument("--curve_path", action="store_true", help="Use curve path navigation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--single_segment", action="store_true", help="Use single segment fast path")

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    os.makedirs(args.save_dir, exist_ok=True)
    pipe = UnifiedLoopConsistencyPipeline(args)
    if args.single_segment:
        pipe.run_single_segment()
    else:
        pipe.run_pipeline()


if __name__ == "__main__":
    main()
