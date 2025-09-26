#!/usr/bin/env python3
"""
Unified Loop Consistency Pipeline

This script combines the entire loop consistency workflow into a single process:
1. Video generation using UNet + SVD (with/without memory)
2. Panoramic to perspective conversion
3. VGGT 3D reconstruction and reprojection

Key optimizations:
- Load generator and reconstructor models once
- Skip intermediate I/O operations by passing data in memory
- Unified error handling and logging
"""

import argparse
import gc
import glob
import math
import os
import sys
from typing import Dict, List, Optional, Tuple, Any
import time

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from equilib import Equi2Pers

import logging

logging.basicConfig(
    level=logging.INFO,  # or DEBUG
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Set environment variables
os.environ["TORCH_HOME"] = "model_cache"
os.environ["HF_HOME"] = "model_cache"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYGLET_HEADLESS"] = "1"
os.environ["PYGLET_BACKEND"] = "egl"

# Add paths
sys.path.append("./evoworld")
sys.path.append("third_party/vggt/")

# Import components
from evoworld.inference.navigator_evoworld import Navigator
from evoworld.inference.forward_evoworld import process_batch
from evoworld.reprojection.pano_to_pers_utils import (
    calculate_segment_indices,
)
from evoworld.trainer.unet_plucker import UNetSpatioTemporalConditionModel
from evoworld.pipeline.pipeline_evoworld import StableVideoDiffusionPipeline

from evoworld.reprojection.reproject_vggt_open3d_utils import (
    SkySegmentationProcessor,
    PointCloudProcessor,
    SceneBuilder,
    CubemapRenderer,
    predictions_to_target_view,
)

from third_party.vggt.vggt.models.vggt import VGGT
from third_party.vggt.vggt.utils.geometry import unproject_depth_map_to_point_map
from third_party.vggt.vggt.utils.load_fn import load_and_preprocess_images
from third_party.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from utils.geometry import xyz_euler_to_four_by_four_matrix_batch
from utils.conversion import numpy_to_image
from utils.plucker_embedding import equirectangular_to_ray, ray_c2w_to_plucker

# Constants
UNITY_TO_OPENCV = [1, -1, 1, -1, 1, -1]


class UnifiedLoopConsistencyPipeline:
    """Unified pipeline for loop consistency evaluation with VGGT reconstruction."""

    def __init__(self, args):
        """Initialize the unified pipeline."""
        self.args = args
        self.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"

        # Initialize models (loaded once)
        self.navigator = None
        self.vggt_processor = None
        self.equi2pers = None

        # Initialize utility processors
        self.point_processor = PointCloudProcessor()
        self.scene_builder = SceneBuilder()
        self.cubemap_renderer = CubemapRenderer()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing unified pipeline...")

    def initialize_models(self):
        """Initialize all models once."""
        self.logger.info("Loading UNet + SVD pipeline...")
        self.navigator = Navigator()
        self.navigator.get_pipeline(
            self.args.unet_path,
            self.args.svd_path,
            model_height=576,
            progress_bar=False,
            num_frames=self.args.num_frames,
        )
        if self.args.single_segment:
            self.logger.info("Loading VGGT model...")
            self.vggt_processor = VGGTProcessor()

        self.logger.info("Initializing Equi2Pers...")
        self.equi2pers = Equi2Pers(
            height=384,  
            width=512,   
            fov_x=90,
            mode="bilinear",
        )

        self.logger.info("All models loaded successfully!")

    def setup_model_and_pipeline(self,args):
        """Setup UNet model and pipeline for inference."""
        weight_dtype = torch.float32

        # Load UNet
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            args.unet_path,
            subfolder="unet",
            low_cpu_mem_usage=True,
        )
        unet.requires_grad_(False)

        # Load pipeline
        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            args.unet_path,
            unet=unet,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)

        # Setup rays for plucker embeddings
        rays = equirectangular_to_ray(target_H=576 // 8, target_W=1024 // 8)
        rays = torch.tensor(rays).to(weight_dtype).to(self.device)

        return pipeline, rays, weight_dtype

    def setup_random_seeds(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)

    def determine_data_config(self) -> Tuple[str, bool]:
        """Determine data root and whether it's a single video."""
        try:
            folder_contents = os.listdir(self.args.base_folder)
            is_single_video = "panorama" in folder_contents
            data_root = self.args.base_folder if is_single_video else f"{self.args.base_folder}/test"
            return data_root, is_single_video
        except FileNotFoundError:
            raise ValueError(f"Base folder '{self.args.base_folder}' not found")

    def create_dataset_and_loader(self, data_root: str, is_single_video: bool):
        """Create dataset and data loader."""
        # Set default dimensions
        self.args.height = 576
        self.args.width = 1024
        sampling_method = 'reprojection' if self.args.single_segment else 'empty_with_traj'
        load_complete_episode = False if self.args.single_segment else True
        if self.args.dataset_name == "JHU":
            from dataset.CameraTrajDataset_JHU import CameraTrajDataset
            dataset = CameraTrajDataset(
                data_root,
                width=self.args.width,
                height=self.args.height,
                trajectory_file=None,
                memory_sampling_args={"sampling_method": "empty_with_traj", "include_initial_frame": True},
                load_complete_episode=load_complete_episode,
                is_single_video=is_single_video,
            )
        elif self.args.dataset_name == "igenex":
            self.args.num_frames = 14
            from dataset.CameraTrajDataset_igenex import CameraTrajDataset
            dataset = CameraTrajDataset(
                data_root,
                width=self.args.width,
                height=self.args.height,
                trajectory_file=None,
                sequence_length=self.args.num_frames,
                last_segment_length=self.args.num_frames,
                memory_sampling_args={"sampling_method": "empty_with_traj", "include_initial_frame": True},
                reprojection_name="rendered_panorama_vggt_open3d",
                load_complete_episode=load_complete_episode,
                is_single_video=is_single_video,
            )
        else:  # Default CameraTrajDataset
            from dataset.CameraTrajDataset import CameraTrajDataset
            
            dataset = CameraTrajDataset(
                data_root,
                width=self.args.width,
                height=self.args.height,
                trajectory_file=None,
                memory_sampling_args={"sampling_method": sampling_method, "include_initial_frame": True},
                load_complete_episode=load_complete_episode,
                is_single_video=is_single_video,
            )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
        )

        return dataset, loader

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor * 0.5 + 0.5
            tensor = tensor.mul(255).byte().detach().cpu().numpy().transpose(1, 2, 0)
            return Image.fromarray(tensor)
        return tensor

    def pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor."""
        if isinstance(image, Image.Image):
            # Apply same transform as dataset
            return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0 * 2 - 1
        return image

    def save_frames(self, frames: List[torch.Tensor], save_path: str, start_idx: int):
        """Save frames to disk."""
        os.makedirs(save_path, exist_ok=True)
        for i, frame in enumerate(frames):
            pil_image = self.tensor_to_pil(frame)
            frame_path = os.path.join(save_path, f"{i+start_idx+1:03}.png")
            pil_image.save(frame_path)

    def process_segment_no_memory(self, batch, episode: str, segment_id: int) -> List[torch.Tensor]:
        """Process a segment without memory (segment 0)."""
        current_path = batch["cam_traj"].to(self.device).squeeze(0)
        images = batch["pixel_values"]

        # Calculate indices
        start_idx = segment_id * (self.args.num_frames - 1)
        end_idx = start_idx + self.args.num_frames

        # Create empty memorized images
        memorized_images = torch.zeros_like(
            batch["memorized_pixel_values"][:, start_idx:end_idx]
        ).to(self.device)

        # Use first frame as start image
        start_image = images[0, 0, :, :, :].to(self.device)

        # Generate navigation
        navigate_fn = getattr(
            self.navigator, "navigate_curve_path" if self.args.curve_path else "navigate_path"
        )
        generations = navigate_fn(
            current_path,
            start_image,
            width=1024,
            height=576,
            num_inference_steps=25,
            memorized_images=memorized_images,
            infer_segment=True,
            segment_id=segment_id,
        )

        navigation = [
            intermediate_image
            for movement in generations
            for intermediate_image in movement
        ]

        return navigation
    
    def convert_Image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor."""
        if isinstance(image, Image.Image):
            # Apply same transform as dataset
            return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0 * 2 - 1
        return image

    def process_segment_with_memory(self, batch, episode: str, segment_id: int, start_image: torch.Tensor,
                                  memorized_pixel_values: List[torch.Tensor]) -> List[torch.Tensor]:
        """Process a segment with memory from previous segment."""
        current_path = batch["cam_traj"].to(self.device).squeeze(0)

        # Use last frame from previous segment as start image
        first_pixel_values = batch["pixel_values"][0,0]
        # insert the first frame of current segment at the beginning of memorized_pixel_values
        memorized_pixel_values.insert(0, first_pixel_values)

        # Create memorized images from previous frames
        memorized_images = torch.stack(memorized_pixel_values).unsqueeze(0).to(self.device)

        # Generate navigation
        navigate_fn = getattr(
            self.navigator, "navigate_curve_path" if self.args.curve_path else "navigate_path"
        )

        generations = navigate_fn(
            current_path,
            start_image,
            width=1024,
            height=576,
            num_inference_steps=25,
            memorized_images=memorized_images,
            infer_segment=True,
            segment_id=segment_id,
        )

        navigation = [
            intermediate_image
            for movement in generations
            for intermediate_image in movement
        ]

        return navigation

    def convert_pano_to_pers(self, panoramic_frames: List[torch.Tensor],
                           camera_params: np.ndarray, segment_id: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """Convert panoramic frames to perspective views."""
        perspective_frames = []
        target_yaws = []

        # Calculate look-at index
        look_at_idx = (segment_id + 1) * 24 + 24

        for i, frame_tensor in enumerate(panoramic_frames):
            # Convert tensor to numpy array for processing
            frame_pil = self.tensor_to_pil(frame_tensor)
            frame_np = np.array(frame_pil)

            # Calculate yaw difference
            current_idx = i +  1
            if current_idx <= len(camera_params):
                current_yaw = camera_params[current_idx-1][4]
                look_at_point = camera_params[min(look_at_idx, len(camera_params)-1)]
                target_yaw_rad = np.arctan2(
                    look_at_point[0] - camera_params[current_idx-1][0],
                    look_at_point[2] - camera_params[current_idx-1][2]
                )
                yaw_diff = current_yaw * np.pi / 180 - target_yaw_rad
            else:
                yaw_diff = 0

            # Convert to perspective
            equi_img = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            equi_img = np.transpose(equi_img, (2, 0, 1))

            pers_img = self.equi2pers(
                equi=equi_img,
                rots={"pitch": 0, "roll": 0, "yaw": yaw_diff},
            )

            pers_img = np.transpose(pers_img, (1, 2, 0))
            perspective_frames.append(pers_img)
            target_yaws.append(yaw_diff / np.pi * 180)

        return perspective_frames, np.array(target_yaws)

    def run_vggt_inference(self, perspective_frames: List[np.ndarray]) -> Dict:
        """Run VGGT inference on perspective frames."""
        # Save perspective frames to temporary files for VGGT preprocessing
        import tempfile
        import os

        temp_dir = tempfile.mkdtemp()
        image_paths = []

        try:
            # Save frames as temporary image files
            for i, frame in enumerate(perspective_frames):
                # Convert BGR to RGB if needed
                if frame.shape[2] == 3:  # Assuming BGR format from OpenCV
                    frame_rgb = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = frame.astype(np.uint8)

                # Save as PIL Image
                pil_image = Image.fromarray(frame_rgb)
                temp_path = os.path.join(temp_dir, f"temp_frame_{i:03d}.png")
                pil_image.save(temp_path)
                image_paths.append(temp_path)

            # Use VGGT's preprocessing function
            images = load_and_preprocess_images(image_paths).to(self.device)

            # Run VGGT inference
            with torch.no_grad():
                with torch.amp.autocast(self.device, dtype=torch.bfloat16):
                    predictions = self.vggt_processor.model(images)

            # Convert pose encoding to extrinsic and intrinsic matrices
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                predictions["pose_enc"], images.shape[-2:]
            )
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic

            # Convert tensors to numpy
            for key in predictions.keys():
                if isinstance(predictions[key], torch.Tensor):
                    predictions[key] = predictions[key].cpu().numpy().squeeze(0)

            # Generate world points from depth map
            depth_map = predictions["depth"]
            world_points = unproject_depth_map_to_point_map(
                depth_map, predictions["extrinsic"], predictions["intrinsic"]
            )
            predictions["world_points_from_depth"] = world_points

            return predictions

        finally:
            # Clean up temporary files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def load_camera_poses(self, episode_path: str) -> np.ndarray:
        """Load camera poses from episode directory."""
        camera_file = os.path.join(episode_path, "camera_poses.txt")
        with open(camera_file, 'r') as f:
            lines = f.readlines()

        # Convert each line to list of floats, skip first line (header) if it exists
        camera_params = []
        for line in lines:
            if line.strip():  # Skip empty lines
                parts = line.strip().split(',')
                if 'Frame' in parts:
                    continue  # Skip header line
                if len(parts) >= 7:  # frame_id + 6 pose parameters
                    camera_params.append([float(x) for x in parts[1:7]])  # Skip frame_id
        camera_params = np.array(camera_params) * UNITY_TO_OPENCV
        return camera_params

    def process_episode(self, batch, episode: str, dataset):
        """Process a complete episode through all segments."""
        self.logger.info(f"\nProcessing episode: {episode}")

        episode_save_dir = os.path.join(self.args.save_dir, episode)
        os.makedirs(episode_save_dir, exist_ok=True)

        # Load actual camera poses from the episode
        episode_path = batch["episode_path"][0]
        camera_params = self.load_camera_poses(episode_path)

        all_generated_frames = []

        # Process each segment
        for segment_id in range(self.args.num_segments):
            self.logger.info(f"Processing segment {segment_id}")
            start_idx, end_idx, look_at_idx = calculate_segment_indices(segment_id)
            if segment_id == 0:
                # First segment (no memory)
                generated_frames = self.process_segment_no_memory(batch, episode, segment_id)
            else:
                current_frame = all_generated_frames[-1]
                current_frame = dataset.transform(current_frame).to(self.device)
                memorized_pixel_values = [dataset.transform(numpy_to_image(memory)) for memory in latest_memories]
                # Subsequent segments (with memory)
                generated_frames = self.process_segment_with_memory(
                    batch, episode, segment_id, current_frame, memorized_pixel_values
                )
            if len(all_generated_frames) > 0:
                generated_frames = generated_frames[1:]  # Avoid duplicating the first frame
            all_generated_frames.extend(generated_frames)

            # Save generated frames if requested
            if self.args.save_frames:
                frames_path = os.path.join(episode_save_dir, f"predictions_{segment_id}")
                start_idx = segment_id * (self.args.num_frames - 1)
                self.save_frames(generated_frames, frames_path, start_idx)

            # Convert to perspective and run VGGT (except for last segment)
            if segment_id < self.args.num_segments - 1:
                self.logger.info(f"Converting segment {segment_id} to perspective...")
                
                # Convert panoramic frames to perspective
                perspective_frames, target_yaws = self.convert_pano_to_pers(
                    all_generated_frames, camera_params, segment_id
                )
                for idx, frame in enumerate(perspective_frames):
                    frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(episode_save_dir, f"perspective_look_at_center_{segment_id}", f"{start_idx+idx+1:03}.png"), frame_bgr)
                tempt_camera_params = camera_params.copy()
                # Update camera parameters with calculated yaw
                if len(target_yaws) > 0:
                    tempt_camera_params[max(0, end_idx-len(target_yaws)):end_idx, 4] = target_yaws

                self.logger.info(f"Running VGGT inference on segment {segment_id}...")

                # Run VGGT inference
                predictions = self.run_vggt_inference(perspective_frames)

                # Generate target views
                self.logger.info(f"Generating target views for segment {segment_id}...")

                # Convert camera params to torch tensor for VGGT
                camera_poses_tensor = torch.tensor(tempt_camera_params, dtype=torch.float32).to(self.device)
                camera_poses_4x4 = xyz_euler_to_four_by_four_matrix_batch(camera_poses_tensor, relative=True)

                outdir = os.path.join(episode_save_dir, f"rendered_panorama_vggt_open3d_{segment_id}")

                latest_memories = predictions_to_target_view(
                    predictions,
                    camera_poses_4x4.cpu().numpy(),
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

                # Clean up memory
                del predictions, perspective_frames
                torch.cuda.empty_cache()

        self.logger.info(f"Episode {episode} processing completed!")
    
    def process_segment(self, batch, episode: str, dataset):
        """
        process last segment of given episode and use the memorized pixel value read from dataset
        """
        self.logger.info(f"\nProcessing episode: {episode}")

        episode_save_dir = os.path.join(self.args.save_dir, episode)
        os.makedirs(episode_save_dir, exist_ok=True)

        # Load actual camera poses from the episode
        episode_path = batch["episode_path"][0]
        camera_params = self.load_camera_poses(episode_path)
        from ipdb import set_trace; set_trace()
        segment_id = self.args.num_segments
        self.logger.info(f"Processing segment {segment_id}")
        start_idx, end_idx, look_at_idx = calculate_segment_indices(segment_id)
        start_image = batch["pixel_values"][0,start_idx].to(self.device)
        memorized_images = batch["memorized_pixel_values"].to(self.device)
        current_path = batch["cam_traj"].to(self.device).squeeze(0)
        navigate_fn = getattr(
            self.navigator, "navigate_curve_path" if self.args.curve_path else "navigate_path"
        )
        from ipdb import set_trace; set_trace()
        with torch.inference_mode():
            generations = navigate_fn(
                current_path,
                start_image,
                width=1024,
                height=576,
                num_inference_steps=25,
                memorized_images=memorized_images,
                infer_segment=True,
                segment_id=segment_id,
        )

        navigation = [
            intermediate_image
            for movement in generations
            for intermediate_image in movement
        ]
        return navigation


    def run_pipeline(self):
        """Run the complete unified pipeline."""

        # Setup
        self.setup_random_seeds()
        self.logger.info(f"Starting unified loop consistency pipeline...")
        self.logger.info(f"UNet Path: {self.args.unet_path}")
        self.logger.info(f"SVD Path: {self.args.svd_path}")

        # Initialize all models once
        self.initialize_models()

        # Determine data configuration
        data_root, is_single_video = self.determine_data_config()

        # Create dataset and loader
        val_dataset, val_loader = self.create_dataset_and_loader(data_root, is_single_video)

        # Process episodes
        for idx, batch in tqdm(enumerate(val_loader)):
            if idx < self.args.start_idx:
                continue
            if idx >= self.args.num_data + self.args.start_idx:
                break

            current_episode = val_dataset.episodes[idx]
            self.process_episode(batch, current_episode, val_dataset)  # camera_params loaded inside
    
    def run_single_segment(self):
        """Run the single segment via unified pipeline."""

        # Setup
        self.setup_random_seeds()
        self.logger.info(f"Starting unified loop consistency pipeline...")
        self.logger.info(f"UNet Path: {self.args.unet_path}")
        self.logger.info(f"SVD Path: {self.args.svd_path}")

        # Initialize all models once
        pipeline, rays, weight_dtype = self.setup_model_and_pipeline(self.args)

        # Determine data configuration
        data_root, is_single_video = self.determine_data_config()

        # Create dataset and loader
        val_dataset, val_loader = self.create_dataset_and_loader(data_root, is_single_video)

        # Process episodes
        for idx, batch in tqdm(enumerate(val_loader)):
            if idx < self.args.start_idx:
                continue
            if idx >= self.args.num_data + self.args.start_idx:
                break

            current_episode = val_dataset.episodes[idx]
            episode_save_dir = os.path.join(self.args.save_dir, current_episode)
            self.args.mask_mem = False # for process_batch
            process_batch(batch, self.args, pipeline, rays, weight_dtype, episode_save_dir, current_episode)





class VGGTProcessor:
    """Lightweight VGGT processor for the unified pipeline."""

    def __init__(self):
        """Initialize VGGT model."""
        self.device = self.device if torch.cuda.is_available() else "cpu"
        self.model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        self.model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        self.model.eval()
        self.model = self.model.to(self.device)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Unified Loop Consistency Pipeline")

    # Model paths
    parser.add_argument("--unet_path", type=str, required=True, help="Path to UNet model")
    parser.add_argument("--svd_path", type=str, default="stabilityai/stable-video-diffusion-img2vid-xt-1-1",
                       help="Path to SVD model")

    # Data configuration
    parser.add_argument("--base_folder", type=str, default="data/Curve_Loop/test",
                       help="Base folder containing episodes")
    parser.add_argument("--save_dir", type=str, default="unified_output",
                       help="Output directory")
    parser.add_argument("--dataset_name", type=str, default="CameraTrajDataset",
                       help="Dataset name")

    # Processing parameters
    parser.add_argument("--num_data", type=int, default=1, help="Number of episodes to process")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index")
    parser.add_argument("--num_segments", type=int, default=3, help="Number of segments to process")
    parser.add_argument("--num_frames", type=int, default=25, help="Frames per segment")
    parser.add_argument("--save_frames", action="store_true", help="Save intermediate frames")

    # Options
    parser.add_argument("--curve_path", action="store_true", help="Use curve path navigation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--single_segment", action="store_true", help="Use single segment")

    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize and run pipeline
    pipeline = UnifiedLoopConsistencyPipeline(args)
    if args.single_segment:
        pipeline.run_single_segment()
    else:
        pipeline.run_pipeline()


if __name__ == "__main__":
    main()
