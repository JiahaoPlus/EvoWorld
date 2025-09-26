# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gc
import glob
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

# Set environment variables
os.environ["TORCH_HOME"] = "model_cache"
os.environ["HF_HOME"] = "model_cache"
os.environ["PYGLET_HEADLESS"] = "1"
os.environ["PYGLET_BACKEND"] = "egl"

# Import utilities
from evoworld.reprojection.reproject_vggt_open3d_utils import (
    predictions_to_glb,
    predictions_to_target_view,
)


sys.path.append("third_party/vggt/")
from third_party.vggt.vggt.models.vggt import VGGT
from third_party.vggt.vggt.utils.geometry import unproject_depth_map_to_point_map
from third_party.vggt.vggt.utils.load_fn import load_and_preprocess_images
from third_party.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from utils.geometry import xyz_euler_to_four_by_four_matrix_batch


class VGGTProcessor:
    """Class for handling VGGT model operations."""

    def __init__(self):
        """Initialize the VGGT model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Check your environment.")

        print("Initializing and loading VGGT model...")
        self.model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        self.model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        self.model.eval()
        self.model = self.model.to(self.device)

    def run_inference(self, target_dir: str, image_subdir: str, only_render_last_24_frame: bool) -> Dict:
        """
        Run the VGGT model on images and return predictions.

        Args:
            target_dir: Directory containing the images
            image_subdir: Subdirectory with images
            only_render_last_24_frame: Whether to exclude last 24 frames

        Returns:
            Dictionary containing model predictions
        """
        print(f"Processing images from {target_dir}")

        # Load and preprocess images
        image_names = glob.glob(os.path.join(target_dir, image_subdir, "*"))
        image_names = sorted(image_names)

        if only_render_last_24_frame:
            image_names = image_names[:-24]

        print(f"Found {len(image_names)} images")
        if len(image_names) == 0:
            raise ValueError("No images found. Check your upload.")

        images = load_and_preprocess_images(image_names).to(self.device)
        print(f"Preprocessed images shape: {images.shape}")

        # Run inference
        print("Running inference...")
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                predictions = self.model(images)

        # Convert pose encoding to extrinsic and intrinsic matrices
        print("Converting pose encoding to extrinsic and intrinsic matrices...")
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
        print("Computing world points from depth map...")
        depth_map = predictions["depth"]
        world_points = unproject_depth_map_to_point_map(
            depth_map, predictions["extrinsic"], predictions["intrinsic"]
        )
        predictions["world_points_from_depth"] = world_points

        # Clean up
        torch.cuda.empty_cache()
        return predictions


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run VGGT demo.")
    parser.add_argument(
        "--data_folder",
        type=str,
        default="data/Curve_Loop/train",
        help="Path to the target directory",
    )
    parser.add_argument(
        "--chunk_num", type=int, default=10, help="Number of chunks to split data into"
    )
    parser.add_argument("--chunk_id", type=int, default=0, help="Chunk ID to process")
    parser.add_argument(
        "--image_subdir",
        type=str,
        default="perspective_look_at_center",
        help="Path to the target directory containing images",
    )
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=50.0,
        help="Confidence threshold for predictions",
    )
    parser.add_argument(
        "--frame_filter", type=str, default="All", help="Frame filter for images"
    )
    parser.add_argument(
        "--mask_black_bg", action="store_true", help="Mask black background"
    )
    parser.add_argument(
        "--mask_white_bg", action="store_true", help="Mask white background"
    )
    parser.add_argument("--show_cam", action="store_true", help="Show camera")
    parser.add_argument("--no_mask_sky", action="store_true", help="Don't mask sky")
    parser.add_argument(
        "--prediction_mode",
        type=str,
        default="Pointmap Regression",
        help="Prediction mode",
    )
    parser.add_argument(
        "--camera_file", type=str, default="camera_poses_look_at_center_0.txt"
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="rendered_panorama_vggt_open3d",
        help="Output subdirectory for rendered panorama",
    )
    parser.add_argument(
        "--num_target_view", type=int, default=24, help="Number of target views"
    )
    parser.add_argument("--only_render_last_24_frame", action="store_true")
    parser.add_argument("--sample_id", type=str, default=None, help="Specific sample ID to process")
    parser.add_argument("--export_glb", action="store_true", help="Export GLB file")
    return parser.parse_args()


def get_target_directories(args) -> List[str]:
    """Get list of target directories to process."""
    input_list = [
        input_dir
        for input_dir in os.listdir(args.data_folder)
        if os.path.isdir(os.path.join(args.data_folder, input_dir))
        and not input_dir.endswith("first_25")
        and not input_dir.endswith("first_75")
    ]
    input_list = sorted(input_list)

    if args.image_subdir in input_list:
        return [""]
    else:
        chunk_size = math.ceil(len(input_list) / args.chunk_num)
        start = args.chunk_id * chunk_size
        end = min((args.chunk_id + 1) * chunk_size, len(input_list))
        target_dirs = input_list[start:end]

        # If sample_id is provided, only process that sample
        if args.sample_id is not None:
            target_dirs = [args.sample_id] if args.sample_id in target_dirs else []

        return target_dirs


def should_skip_processing(target_dir: str, output_subdir: str, num_target_view: int) -> bool:
    """Check if processing should be skipped because output already exists."""
    output_dir_check = os.path.join(target_dir, output_subdir)
    if not os.path.exists(output_dir_check):
        return False

    image_count = len([f for f in os.listdir(output_dir_check) if f.endswith(".png")])
    return image_count == num_target_view


def load_camera_poses(pose_file: str) -> torch.Tensor:
    """Load camera poses from file."""
    with open(pose_file) as f:
        raw_pose = f.readlines()

    raw_pose = [list(map(float, x.strip().split(" ")))[1:] for x in raw_pose]
    raw_pose = torch.tensor(raw_pose).to("cuda")
    poses = xyz_euler_to_four_by_four_matrix_batch(raw_pose, relative=True)
    return poses


def perform_reconstruction(
    processor: VGGTProcessor,
    target_dir: str,
    args,
    camera_pose: torch.Tensor
):
    """Perform reconstruction using VGGT predictions."""
    if not os.path.isdir(target_dir) or target_dir == "None":
        print(f"No valid target directory found: {target_dir}")
        return

    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache()

    print("Running model inference...")
    predictions = processor.run_inference(
        target_dir, args.image_subdir, args.only_render_last_24_frame
    )

    outdir = os.path.join(target_dir, args.output_subdir)

    # Export GLB if requested
    if args.export_glb:
        glbfile = os.path.join(
            target_dir,
            f"glbscene_{args.conf_thres}_{args.frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{args.mask_black_bg}_maskw{args.mask_white_bg}_cam{args.show_cam}_sky{args.mask_sky}_pred{args.prediction_mode.replace(' ', '_')}.glb",
        )

        glbscene = predictions_to_glb(
            predictions,
            conf_thres=args.conf_thres,
            filter_by_frames=args.frame_filter,
            mask_black_bg=args.mask_black_bg,
            mask_white_bg=args.mask_white_bg,
            show_cam=args.show_cam,
            mask_sky=args.mask_sky,
            target_dir=target_dir,
            image_subdir=args.image_subdir,
            prediction_mode=args.prediction_mode,
            only_render_last_24_frame=args.only_render_last_24_frame,
        )
        glbscene.export(file_obj=glbfile)
        print(f"GLB file saved at: {glbfile}")

    # Generate target views
    predictions_to_target_view(
        predictions,
        camera_pose,
        conf_thres=args.conf_thres,
        filter_by_frames=args.frame_filter,
        mask_black_bg=args.mask_black_bg,
        mask_white_bg=args.mask_white_bg,
        show_cam=args.show_cam,
        mask_sky=args.mask_sky,
        target_dir=target_dir,
        image_subdir=args.image_subdir,
        prediction_mode=args.prediction_mode,
        num_target_view=args.num_target_view,
        outdir=outdir,
        only_render_last_24_frame=args.only_render_last_24_frame,
    )

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()


def main():
    """Main function for VGGT reconstruction."""
    args = parse_arguments()

    # Set mask_sky based on no_mask_sky flag
    args.mask_sky = not args.no_mask_sky

    # Get target directories to process
    target_dir_list = get_target_directories(args)
    print(f"Total number of target directories: {len(target_dir_list)}")

    # Initialize VGGT processor
    processor = VGGTProcessor()

    # Process each target directory
    for target_dir_name in tqdm(target_dir_list):
        target_dir = os.path.join(args.data_folder, target_dir_name)

        # Check if processing should be skipped
        if should_skip_processing(target_dir, args.output_subdir, args.num_target_view):
            tqdm.write(f"Skipping {target_dir} as it already contains {args.num_target_view} images.")
            continue

        # Load camera poses
        pose_file = os.path.join(target_dir, args.camera_file)
        camera_pose = load_camera_poses(pose_file)

        # Perform reconstruction
        perform_reconstruction(processor, target_dir, args, camera_pose)


if __name__ == "__main__":
    main()
