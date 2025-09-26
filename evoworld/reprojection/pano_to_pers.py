# This script is modified from 360-1M:https://github.com/MattWallingford/360-1M/blob/main/VideoProcessing/video_to_frames.py
import argparse
import os
from typing import List

import cv2
import numpy as np
from equilib import Equi2Pers

UNITY_TO_OPENCV = [1, -1, 1, -1, 1, -1]


def parse_arguments():
    """Parse command-line arguments for panoramic to perspective conversion."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder",
        type=str,
        default="/home/jwang384/scratchayuille1/jwang384/Genex_mem/data/debug/test/episode_0001/panorama",
        help="Path to video folder",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="/home/jwang384/scratchayuille1/jwang384/Genex_mem/third_party/360-1M/VideoProcessing/extracted_frames_from_png_gt",
        help="Path to the output folder",
    )
    parser.add_argument(
        "--fov",
        type=int,
        default=90,
        help="FOV of the perspective images",
    )
    parser.add_argument(
        "--camera_file",
        type=str,
        required=True,
        help="Path to camera parameters file"
    )
    parser.add_argument(
        "--output_camera_file",
        type=str,
        required=True,
        help="Path to output camera parameters file"
    )
    return parser.parse_args()


def read_camera_file_and_convert_to_rdf(camera_file: str) -> np.ndarray:
    """Read camera parameters file and convert to RDF format."""
    with open(camera_file, 'r') as f:
        lines = f.readlines()

    # Convert each line to list of floats, skip first line (header)
    camera_params = [list(map(float, line.strip().split(',')))[1:] for line in lines[1:]]
    camera_params = np.array(camera_params)

    # Convert to RDF
    camera_params = camera_params * UNITY_TO_OPENCV
    return camera_params


def write_camera_file(camera_params: np.ndarray, output_camera_file: str):
    """Write camera parameters to file."""
    with open(output_camera_file, 'w') as f:
        for i in range(len(camera_params)):
            f.write(f"{i+1} {camera_params[i][0]} {camera_params[i][1]} {camera_params[i][2]} "
                   f"{camera_params[i][3]} {camera_params[i][4]} {camera_params[i][5]}\n")


def calculate_target_yaw_last_segment(current_idx: int, camera_params: np.ndarray) -> float:
    """Calculate target yaw for the last segment."""
    target_yaw = camera_params[-1][4]
    current_yaw = camera_params[current_idx-1][4]
    yaw_diff = (current_yaw - target_yaw) / 180 * np.pi
    return yaw_diff


def calculate_target_yaw(current_idx: int, camera_params: np.ndarray) -> float:
    """Calculate target yaw for regular segments."""
    current_yaw = camera_params[current_idx-1][4]
    look_at_point = camera_params[-1]
    target_yaw = np.arctan2(
        look_at_point[0] - camera_params[current_idx-1][0],
        look_at_point[2] - camera_params[current_idx-1][2]
    )
    yaw_diff = current_yaw * np.pi / 180 - target_yaw
    return yaw_diff


def process_single_image(
    data_path: str,
    output_folder: str,
    equi2pers: Equi2Pers,
    camera_params: np.ndarray,
    is_last_segment: bool = False
) -> float:
    """Process a single panoramic image to perspective."""
    # Extract frame index from path
    current_idx = int(data_path.split(".")[0].split('/')[-1])

    # Calculate yaw difference based on segment type
    if is_last_segment:
        yaw_diff = calculate_target_yaw_last_segment(current_idx, camera_params)
        target_yaw = camera_params[-1][4]
    else:
        yaw_diff = calculate_target_yaw(current_idx, camera_params)
        # Convert back to degrees for return value
        target_yaw = (yaw_diff + calculate_target_yaw(current_idx, camera_params) - yaw_diff) / np.pi * 180

    # Read and convert image
    img = cv2.imread(data_path)
    equi_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    equi_img = np.transpose(equi_img, (2, 0, 1))

    # Convert to perspective
    pers_img = equi2pers(
        equi=equi_img,
        rots={"pitch": 0, "roll": 0, "yaw": yaw_diff},
    )

    # Save output
    output_path = os.path.join(output_folder, f"frame_{current_idx:03d}.png")
    pers_img = np.transpose(pers_img, (1, 2, 0))
    pers_img_bgr = cv2.cvtColor(pers_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, pers_img_bgr)

    return target_yaw


def collect_image_paths(data_folder: str) -> List[str]:
    """Collect all PNG image paths from the data folder."""
    data_list = sorted(os.listdir(data_folder))
    image_path_list = [
        os.path.join(data_folder, img_file)
        for img_file in data_list
        if img_file.endswith(".png")
    ]
    return image_path_list


def process_videos(
    args,
    data_folder: str,
    output_folder: str,
    equi2pers: Equi2Pers,
    camera_params: np.ndarray
) -> np.ndarray:
    """Process all videos/images in the data folder."""
    os.makedirs(output_folder, exist_ok=True)

    # Collect all image paths
    image_path_list = collect_image_paths(data_folder)
    total_frames = len(image_path_list)
    print(f"Total frames: {total_frames}")

    # Process each image
    target_yaw = []
    for i, img_path in enumerate(image_path_list):
        # Determine if this is the last segment (last 25 frames)
        is_last_segment = i > total_frames - 25

        t_yaw = process_single_image(
            img_path, output_folder, equi2pers, camera_params, is_last_segment
        )
        target_yaw.append(t_yaw)

    return np.array(target_yaw)


def main():
    """Main function for panoramic to perspective conversion."""
    args = parse_arguments()

    # Initialize Equi2Pers
    equi2pers = Equi2Pers(
        height=384,
        width=512,
        fov_x=args.fov,
        mode="bilinear",
    )

    # Read camera parameters
    camera_params = read_camera_file_and_convert_to_rdf(args.camera_file)

    # Process videos
    target_yaw = process_videos(
        args, args.data_folder, args.output_folder, equi2pers, camera_params
    )

    # Update camera parameters with calculated yaw
    if len(target_yaw) > 0:
        camera_params[:, 4] = target_yaw

    # Write updated camera parameters
    write_camera_file(camera_params, args.output_camera_file)


# use env "genex_mem" to run this script
if __name__ == "__main__":
    main()
