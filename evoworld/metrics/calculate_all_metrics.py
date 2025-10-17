import numpy as np
import torch
from tqdm import tqdm
import os
import cv2
from torchvision.utils import save_image
from evoworld.metrics.other_metrics.calculate_lpips import calculate_lpips
from evoworld.metrics.other_metrics.calculate_ssim import calculate_ssim
from evoworld.metrics.other_metrics.calculate_psnr import calculate_psnr
from evoworld.metrics.other_metrics.calculate_latent_mse import calculate_latent_mse


def to_BCTHW(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4)

    return x


def calculate_fvd(videos1, videos2, device, method="styleganv"):

    if method == "styleganv":
        from evoworld.metrics.fvd.styleganv.fvd import (
            get_fvd_feats,
            frechet_distance,
            load_i3d_pretrained,
        )
    elif method == "videogpt":
        from evoworld.metrics.fvd.videogpt.fvd import load_i3d_pretrained
        from evoworld.metrics.fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats
        from evoworld.metrics.fvd.videogpt.fvd import frechet_distance

    print("calculate_fvd...")

    # videos [batch_size, timestamps, channel, h, w]

    assert videos1.shape == videos2.shape

    i3d = load_i3d_pretrained(device=device)

    print("i3d loaded")
    fvd_results = []

    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]

    videos1 = to_BCTHW(videos1)
    videos2 = to_BCTHW(videos2)

    fvd_results = {}

    # for calculate FVD, each clip_timestamp must >= 10
    for clip_timestamp in tqdm(range(10, videos1.shape[-3] + 1)):

        # get a video clip
        # videos_clip [batch_size, channel, timestamps[:clip], h, w]
        videos_clip1 = videos1[:, :, :clip_timestamp]
        videos_clip2 = videos2[:, :, :clip_timestamp]
        print("videos_clip1.shape", videos_clip1.shape)
        # get FVD features
        feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
        feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)

        # calculate FVD when timestamps[:clip]
        fvd_results[clip_timestamp] = frechet_distance(feats1, feats2)

    result = {
        "value": fvd_results,
        "video_setting": videos1.shape,
        "video_setting_name": "batch_size, channel, time, heigth, width",
    }

    return result


def calculate_fvd_batch(videos1, videos2, device, method="styleganv", batch_size=10):
    """
    Compute FVD in a memory-efficient manner using batch processing.

    Args:
        videos1 (torch.Tensor): First set of videos [batch_size, timestamps, channel, h, w]
        videos2 (torch.Tensor): Second set of videos [batch_size, timestamps, channel, h, w]
        device (str): Device to run the computation on ('cuda' or 'cpu').
        method (str): Method to use for FVD calculation ('styleganv' or 'videogpt').
        batch_size (int): Number of videos to process at a time.

    Returns:
        dict: FVD values at different timestamps.
    """

    if method == "styleganv":
        from evoworld.metrics.fvd.styleganv.fvd import (
            get_fvd_feats,
            frechet_distance,
            load_i3d_pretrained,
        )
    elif method == "videogpt":
        from evoworld.metrics.fvd.videogpt.fvd import load_i3d_pretrained
        from evoworld.metrics.fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats
        from evoworld.metrics.fvd.videogpt.fvd import frechet_distance

    print("calculate_fvd...")

    assert videos1.shape == videos2.shape, "Video sets must have the same shape."

    i3d = load_i3d_pretrained(device=device)
    print("i3d loaded")

    fvd_results = {}

    # Support grayscale input (convert grayscale to 3-channel RGB)
    videos1 = to_BCTHW(videos1)
    videos2 = to_BCTHW(videos2)

    num_videos = videos1.shape[0]

    # Process each clip_timestamp >= 10
    for clip_timestamp in tqdm(range(10, videos1.shape[-3] + 1)):
        fvd_feats1 = []
        fvd_feats2 = []

        # Process videos in batches to prevent OOM
        for start_idx in range(0, num_videos, batch_size):
            end_idx = min(start_idx + batch_size, num_videos)

            # Extract batch
            videos_clip1 = videos1[start_idx:end_idx, :, :clip_timestamp]
            videos_clip2 = videos2[start_idx:end_idx, :, :clip_timestamp]

            print(
                f"Processing batch {start_idx} to {end_idx}, shape: {videos_clip1.shape}"
            )

            # Extract FVD features
            feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
            feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)

            # Collect features for later distance calculation
            fvd_feats1.append(torch.tensor(feats1))
            fvd_feats2.append(torch.tensor(feats2))

        # Concatenate collected features across all batches
        fvd_feats1 = torch.cat(fvd_feats1, dim=0)
        fvd_feats2 = torch.cat(fvd_feats2, dim=0)

        # Compute FVD for this timestamp
        fvd_results[clip_timestamp] = frechet_distance(fvd_feats1, fvd_feats2)

    return {
        "value": fvd_results,
        "value_mean": float(np.mean(list(fvd_results.values()))),
        "fvd_setting": method,
        "video_setting": videos1.shape,
        "video_setting_name": "batch_size, channel, time, height, width",
    }


def read_video_our(data_path, subdir, ground_truth=True, num_videos=None):
    """
    read video from dir
    Args:
        video_path: video dir
        ground_truth: if True, read GT video, else read generated video
    Return:
        video: [batch_size, timestamps, channel, h, w]
    """
    episode_subfolder = sorted(
        d for d in os.listdir(args.data_path)
        if os.path.isdir(os.path.join(args.data_path, d))
    )
    if num_videos:
        episode_subfolder = episode_subfolder[:num_videos]

    frames_all_videos = []
    for episode in episode_subfolder:
        video_path = os.path.join(data_path, episode, subdir)
        frame_names = sorted(os.listdir(video_path))[-25:]  # TODO: only to_BCTHWfer the last 25 frames to video
        frames = []
        for frame_name in frame_names:
            frame_path = os.path.join(video_path, frame_name)
            # read image
            frame = cv2.imread(frame_path)
            frames.append(frame)
        frames_all_videos.append(np.array(frames))

    frames_all_videos = torch.tensor(np.array(frames_all_videos)).permute(0, 1, 4, 2, 3)
    print("frames_all_videos.shape", frames_all_videos.shape)
    return frames_all_videos

def main(args):
    # NUMBER_OF_VIDEOS = 100
    VIDEO_LENGTH = 25
    CHANNEL = 3
    # SIZE = 64
    SIZE = 224

    gt_videos = read_video_our(args.data_path, args.gt_subdir, ground_truth=True, num_videos=args.num_videos)
    gen_videos = read_video_our(args.data_path, args.gen_subdir, ground_truth=False, num_videos=args.num_videos)

    _, _, C, H, W = gt_videos.shape
    # normalize to [0, 1]
    video1 = gt_videos / 255.0
    video2 = gen_videos / 255.0
    print("video 1 range: ", video1.max(), video1.min())
    print("video 2 range: ", video2.max(), video2.min())

    device = torch.device("cuda")

    import json
    result = {}
    result['fvd'] = calculate_fvd_batch(video1, video2, device, method="styleganv")
    result['ssim'] = calculate_ssim(video1, video2)
    result['psnr'] = calculate_psnr(video1, video2)
    result['lpips'] = calculate_lpips(video1, video2, device)
    result['latent_mse'] = calculate_latent_mse(video1, video2)
    result['loop_closure_latent_mse'] = calculate_latent_mse(video1[:, -1:], video2[:, -1:])

    print(json.dumps(result, indent=4))

    if args.result_file:
        with open(args.result_file, "w") as f:
            json.dump(result, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/Segment_Consistency/test",
    )
    parser.add_argument(
        "--gt_subdir",
        type=str,
        default="predictions_gt_1",
    )
    parser.add_argument(
        "--gen_subdir",
        type=str,
        default="predictions_1",
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default="eval_score.json",
    )
    parser.add_argument("--num_videos", type=int, default=100)
    parser.add_argument("--test_length", type=int, default=25)
    args = parser.parse_args()
    args.result_file = os.path.join(args.data_path, args.result_file)
    os.makedirs(os.path.dirname(args.result_file), exist_ok=True)
    main(args)
