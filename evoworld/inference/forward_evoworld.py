import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dataset.CameraTrajDataset import (CameraTrajDataset,
                                       xyz_euler_to_three_by_four_matrix_batch)
from evoworld.pipeline.pipeline_evoworld import StableVideoDiffusionPipeline
from evoworld.trainer.unet_plucker import UNetSpatioTemporalConditionModel
from utils.plucker_embedding import equirectangular_to_ray, ray_c2w_to_plucker

sys.path.append("./evoworld")


def parse_arguments():
    """Parse command-line arguments for forward pass evaluation."""
    parser = argparse.ArgumentParser(description="Read safetensors file")
    parser.add_argument("--ckpt", type=str, help="Path to safetensors file")
    parser.add_argument("--step_id", type=str, help="e.g., 20000")
    parser.add_argument("--data", type=str, help="Path to data file")
    parser.add_argument("--width", type=int, help="Width of the image")
    parser.add_argument("--height", type=int, help="Height of the image")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index of the image")
    parser.add_argument("--num_data", type=int, default=1, help="Num of images to test.")
    parser.add_argument(
        "--add_plucker", action="store_true", help="Whether to add plucker embeddings"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Whether to enable verbose logging"
    )
    parser.add_argument("--mask_mem", action="store_true")
    parser.add_argument("--num_frames", default=25, type=int)
    parser.add_argument(
        "--reprojection_name", type=str, default="rendered_panorama_vggt_open3d"
    )
    parser.add_argument("--output_name", type=str, default="eval_add_mem")
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def setup_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def determine_data_config(data_path: str, reprojection_name: str) -> Tuple[str, bool]:
    """Determine data root and whether it's a single video."""
    try:
        folder_contents = os.listdir(data_path)
        is_single_video = reprojection_name in folder_contents
        data_root = data_path if is_single_video else f"{data_path}/test"
        return data_root, is_single_video
    except FileNotFoundError:
        raise ValueError(f"Data path '{data_path}' not found")


def create_dataset_and_loader(args, data_root: str, is_single_video: bool, loop_args: dict):
    """Create dataset and data loader for evaluation."""
    val_dataset = CameraTrajDataset(
        data_root,
        width=args.width,
        height=args.height,
        trajectory_file=None,
        pos_scale=1.0,
        memory_sampling_args=loop_args,
        sequence_length=args.num_frames,
        last_segment_length=args.num_frames,
        reprojection_name=args.reprojection_name,
        is_single_video=is_single_video,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
    )

    return val_dataset, val_loader


def setup_model_and_pipeline(args):
    """Setup UNet model and pipeline for inference."""
    weight_dtype = torch.float32

    # Load UNet
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.ckpt,
        subfolder="unet",
        low_cpu_mem_usage=True,
    )
    unet.requires_grad_(False)

    # Load pipeline
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        args.ckpt,
        unet=unet,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=not args.verbose)

    # Setup rays for plucker embeddings
    rays = equirectangular_to_ray(target_H=args.height // 8, target_W=args.width // 8)
    rays = torch.tensor(rays).to(weight_dtype).to("cuda")

    return pipeline, rays, weight_dtype


def prepare_batch_data(batch, args, rays, weight_dtype):
    """Prepare batch data including camera trajectories and plucker embeddings."""
    images = batch["pixel_values"]
    first_frame = images[:, 0, :, :, :].to("cuda")
    camera_traj_raw = batch["cam_traj"].to("cuda")

    # Initialize camera trajectory tensor
    camera_traj = (
        torch.zeros(camera_traj_raw.shape[0], args.num_frames, 3, 4)
        .to(weight_dtype)
        .to("cuda", non_blocking=True)
    )

    # Initialize plucker embedding tensor
    plucker_embedding = (
        torch.zeros(
            camera_traj_raw.shape[0],
            args.num_frames,
            6,
            args.height // 8,
            args.width // 8,
        )
        .to(weight_dtype)
        .to("cuda", non_blocking=True)
    )

    # Compute camera trajectories and plucker embeddings
    for i in range(camera_traj.shape[0]):
        camera_traj[i] = xyz_euler_to_three_by_four_matrix_batch(
            camera_traj_raw[i], relative=True
        )  # Step, 3, 4
        plucker_embedding[i] = ray_c2w_to_plucker(
            rays, camera_traj[i]
        )  # Step, 6, 72, 128

    memorized_pixel_values = batch["memorized_pixel_values"].to("cuda")

    return first_frame, camera_traj, plucker_embedding, memorized_pixel_values, images


def save_frames(video_frames, gt_frames, frames_path: str, frames_gt_path: str, num_frames: int):
    """Save predicted and ground truth frames to disk."""
    os.makedirs(frames_path, exist_ok=True)
    os.makedirs(frames_gt_path, exist_ok=True)

    assert (
        len(video_frames) == num_frames
    ), f"video frames {len(video_frames)} should equal num_frames {num_frames}!"

    for i in range(num_frames):
        frame = video_frames[i]
        gt_frame = gt_frames[i]

        # Convert ground truth frame to PIL
        gt_frame = gt_frame * 0.5 + 0.5
        gt_frame = Image.fromarray(
            gt_frame.mul(255).byte().detach().cpu().numpy().transpose(1, 2, 0)
        )

        # Save frames
        frame.save(os.path.join(frames_path, f"{i+1:03}.png"))
        gt_frame.save(os.path.join(frames_gt_path, f"{i+1:03}.png"))


def process_batch(batch, args, pipeline, rays, weight_dtype, output_path: str, episode: str):
    """Process a single batch for inference and save results."""
    # Prepare batch data
    first_frame, camera_traj, plucker_embedding, memorized_pixel_values, images = prepare_batch_data(
        batch, args, rays, weight_dtype
    )

    # Run inference
    with torch.inference_mode():
        video_frames = pipeline(
            first_frame,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            decode_chunk_size=8,
            motion_bucket_id=127,
            fps=7,
            noise_aug_strength=0.02,
            plucker_embedding=plucker_embedding,
            memorized_pixel_values=memorized_pixel_values,
            mask_mem=args.mask_mem,
        ).frames[0]

    # Setup output paths
    frames_path = os.path.join(output_path, episode, "predictions")
    frames_gt_path = os.path.join(output_path, episode, "predictions_gt")

    # Save frames
    save_frames(video_frames, images[0], frames_path, frames_gt_path, args.num_frames)


def main():
    """Main function for forward pass evaluation."""
    args = parse_arguments()

    # Setup
    setup_random_seeds(args.seed)

    # Configuration
    loop_args = {
        "sampling_method": "reprojection",
        "include_initial_frame": True,
    }

    # Setup output path
    output_path = os.path.join(args.ckpt, args.output_name)
    os.makedirs(output_path, exist_ok=True)

    # Determine data configuration
    data_root, is_single_video = determine_data_config(args.data, args.reprojection_name)

    # Create dataset and loader
    val_dataset, val_loader = create_dataset_and_loader(args, data_root, is_single_video, loop_args)

    # Setup model and pipeline
    pipeline, rays, weight_dtype = setup_model_and_pipeline(args)

    # Process batches
    for idx, batch in tqdm(enumerate(val_loader)):
        if idx < args.start_idx:
            continue
        if idx >= args.num_data + args.start_idx:
            break

        current_episode = val_dataset.episodes[idx]


        process_batch(batch, args, pipeline, rays, weight_dtype, output_path, current_episode)



if __name__ == "__main__":
    main()
