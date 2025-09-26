# Utility functions for training script

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


# copy from https://github.com/crowsonkb/k-diffusion.git
def rand_log_normal(shape, loc=0.0, scale=1.0, device="cpu", dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


def _replace_unet_conv_in_zero_init(unet, n_cond=2, n_memory=0, add_plucker=False):
    _plucker_dim = 6 if add_plucker else 0
    _in_channels = 4 * (1 + n_cond + n_memory) + _plucker_dim

    # Clone original weights and biases
    _weight = unet.conv_in.weight.clone()
    _bias = unet.conv_in.bias.clone()

    # Define shared parameters
    _weight_append = nn.Parameter(torch.zeros_like(_weight[:, -4:, :, :]))
    _weight_memories = nn.Parameter(torch.zeros_like(_weight[:, -4:, :, :]))
    _weight_plucker = (
        nn.Parameter(torch.zeros_like(_weight[:, :_plucker_dim, :, :]))
        if add_plucker
        else None
    )

    # Add conditions
    for _ in range(n_cond - 1):
        _weight = torch.cat([_weight, _weight_append], dim=1)

    # Add memory channels
    for _ in range(n_memory):
        _weight = torch.cat([_weight, _weight_memories], dim=1)

    # Add plucker weights if applicable
    if add_plucker:
        _weight = torch.cat([_weight, _weight_plucker], dim=1)

    assert _weight.shape[1] == _in_channels, "Mismatch in input channel dimensions"

    # Create new conv layer with modified weights
    _n_convin_out_channel = unet.conv_in.out_channels
    _new_conv_in = nn.Conv2d(
        _in_channels,
        _n_convin_out_channel,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
    )
    _new_conv_in.weight = nn.Parameter(_weight)
    _new_conv_in.bias = nn.Parameter(_bias)

    # Replace the layer and update config
    unet.conv_in = _new_conv_in
    unet.config["in_channels"] = _in_channels

    return


# resizing utils
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners
    )
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1
    )

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (
        torch.arange(window_size, device=sigma.device, dtype=sigma.dtype)
        - window_size // 2
    ).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


def export_to_video(video_frames, output_video_path, fps):
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    from PIL import Image

    # Convert numpy arrays to PIL Images if needed
    pil_frames = [
        Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
        for frame in frames
    ]

    pil_frames[0].save(
        output_gif_path.replace(".mp4", ".gif"),
        format="GIF",
        append_images=pil_frames[1:],
        save_all=True,
        duration=500,
        loop=0,
    )


def tensor_to_vae_latent(t, vae):
    from einops import rearrange

    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents


def print_cuda_memory_usage(cuda_index=0, show=False):
    """
    Prints the memory usage of a specified CUDA device.

    Args:
        cuda_index (int): The index of the CUDA device to check. Defaults to 0.
    """
    # Check if CUDA is available and the device is initialized
    if torch.cuda.is_available():
        # Get current memory usage for the specified CUDA device
        current_memory_allocated = torch.cuda.memory_allocated(cuda_index)
        current_memory_reserved = torch.cuda.memory_reserved(cuda_index)

        # Convert to MB
        current_memory_allocated_MB = current_memory_allocated / 1024**2
        current_memory_reserved_MB = current_memory_reserved / 1024**2

        if show:
            # Print memory usage
            print("-" * 50)
            print(f"CUDA Device {cuda_index}:")
            print(f"  Allocated Memory: {current_memory_allocated_MB:.2f} MB")
            print(f"  Reserved Memory: {current_memory_reserved_MB:.2f} MB")
            print("-" * 50, flush=True)
        return current_memory_allocated_MB, current_memory_reserved_MB
    else:
        print("CUDA is not available.")


def print_tensor_vram_usage(tensor):
    """
    Prints the VRAM usage of a specific PyTorch tensor in gigabytes (GB).

    Args:
        tensor (torch.Tensor): The input tensor to measure VRAM usage for.
    """
    if not tensor.is_cuda:
        print("The tensor is not on a CUDA device.")
        return

    # Calculate the VRAM usage in bytes
    vram_usage_bytes = tensor.element_size() * tensor.nelement()

    # Convert to gigabytes (GB)
    vram_usage_gb = vram_usage_bytes / (1024**3)

    print(f"VRAM usage of the tensor: {vram_usage_gb:.6f} GB")


def download_image(url):
    from urllib.parse import urlparse

    from diffusers.utils import load_image
    from PIL import Image
    original_image = (
        lambda image_url_or_path: load_image(image_url_or_path)
        if urlparse(image_url_or_path).scheme
        else Image.open(image_url_or_path).convert("RGB")
    )(url)
    return original_image
