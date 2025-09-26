#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# this version has unfreeze conv in and conv out in unet

"""Script to fine-tune Stable Video Diffusion."""
import argparse
import logging
import math
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

import accelerate
import cv2
import diffusers
import numpy as np
import PIL
import torch
import torch.nn as nn
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKLTemporalDecoder
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from einops import rearrange
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torch.utils.data import RandomSampler
from torchvision.transforms.functional import resize
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from dataset.CameraTrajDataset import (
    CameraTrajDataset,
    xyz_euler_to_three_by_four_matrix_batch,
)
from evoworld.pipeline.pipeline_evoworld import (
    StableVideoDiffusionPipeline,
    convert_pano_to_mono,
)
from evoworld.trainer.unet_plucker import UNetSpatioTemporalConditionModel
from evoworld.trainer.arg_parser import parse_args
from evoworld.trainer.trainer_utils import (
    rand_log_normal,
    _replace_unet_conv_in_zero_init,
    _resize_with_antialiasing,
    _compute_padding,
    _filter2d,
    _gaussian,
    _gaussian_blur2d,
    export_to_video,
    export_to_gif,
    tensor_to_vae_latent,
    print_cuda_memory_usage,
    print_tensor_vram_usage,
    download_image,
)
from utils.plucker_embedding import equirectangular_to_ray, ray_c2w_to_plucker

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")
logger = get_logger(__name__, log_level="INFO")
PROJ_NAME = "evoworld"


def main():
    loop_args = {
        "sampling_method": "reprojection",
        "num_memories": 1,
        "include_initial_frame": True,
    }
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    logger.info(f"Number of visible GPUs: {torch.cuda.device_count()}")
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(
            os.environ.get("WANDB_PROJECT", PROJ_NAME),
            config=vars(args),
            init_kwargs={"wandb": {"name": os.environ.get("WANDB_RUN_NAME", None)}},
        )

    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} name: {torch.cuda.get_device_name(i)}")

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # Load img encoder, tokenizer and models.
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="feature_extractor",
        revision=args.revision,
    )

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="image_encoder",
        revision=args.revision,
    )

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant="fp16",
    )

    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        (
            args.pretrained_model_name_or_path
            if args.pretrain_unet is None
            else args.pretrain_unet
        ),
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16",
    )
    if "stable-video-diffusion" in args.pretrained_model_name_or_path:
        _replace_unet_conv_in_zero_init(
            unet,
            n_cond=1,
            n_memory=loop_args["num_memories"],
            add_plucker=args.add_plucker,
        )

    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    unet.accelerator = accelerator

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move image_encoder and vae to gpu and cast to weight_dtype
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    logger.info(f"Accelerate Device: {accelerator.device} {'*' * 20}")
    vae.to(accelerator.device, dtype=weight_dtype)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(
            unet.parameters(),
            model_cls=UNetSpatioTemporalConditionModel,
            model_config=unet.config,
        )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))
                if weights:  # Don't pop if empty
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"),
                    UNetSpatioTemporalConditionModel,
                )
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNetSpatioTemporalConditionModel.from_pretrained(
                    input_dir, subfolder="unet"
                )
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.per_gpu_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    parameters_list = []

    # Customize the parameters that need to be trained;
    for name, param in unet.named_parameters():
        if (
            "temporal_transformer_block" in name
            or "conv_in" in name
            or "conv_out" in name
            or "norm" in name
            or "Norm" in name
        ):
            parameters_list.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
    logger.info(f"Trainable params num: {len(parameters_list)}")

    optimizer = optimizer_cls(
        parameters_list,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # DataLoaders creation:
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes

    train_dataset = CameraTrajDataset(
        f"{args.base_folder}/train",
        width=args.width,
        height=args.height,
        trajectory_file=None,
        memory_sampling_args=loop_args,
        last_segment_length=args.num_frames,
        reprojection_name=args.reprojection_name,
    )

    val_dataset = CameraTrajDataset(
        f"{args.base_folder}/val",
        width=args.width,
        height=args.height,
        trajectory_file=None,
        memory_sampling_args=loop_args,
        last_segment_length=args.num_frames,
        reprojection_name=args.reprojection_name,
    )

    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
    )
    if accelerator.is_main_process:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=args.num_workers,
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        unet, optimizer, lr_scheduler, train_dataloader
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # attribute handling for models using DDP
    if isinstance(
        unet, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
    ):
        unet = unet.module

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    

    # Train!
    total_batch_size = (
        args.per_gpu_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    def encode_image(pixel_values):
        # pixel: [-1, 1]
        pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
        # We unnormalize it after resizing.
        pixel_values = (pixel_values + 1.0) / 2.0

        # Normalize the image with for CLIP input
        pixel_values = feature_extractor(
            images=pixel_values,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        pixel_values = pixel_values.to(device=accelerator.device, dtype=weight_dtype)
        image_embeddings = image_encoder(pixel_values).image_embeds
        return image_embeddings

    def _get_add_time_ids(
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
        if hasattr(unet, "module"):
            passed_add_embed_dim = unet.module.config.addition_time_embed_dim * len(
                add_time_ids
            )
        else:
            passed_add_embed_dim = unet.config.addition_time_embed_dim * len(
                add_time_ids
            )
        expected_add_embed_dim = unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        return add_time_ids

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    logger.info(f"arguments: ")
    logger.info(args)
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    # get ray direction for each pixel: H, W, 3
    rays = equirectangular_to_ray(
        target_H=args.height // 8, target_W=args.width // 8
    )
    rays = torch.tensor(rays).to(weight_dtype).to(accelerator.device, non_blocking=True)
    num_frames = args.num_frames
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            torch.cuda.empty_cache()
            # Skip steps until we reach the resumed step
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # first, convert images to latent space.
                pixel_values = (
                    batch["pixel_values"]
                    .to(weight_dtype)
                    .to(accelerator.device, non_blocking=True)
                )
                camera_traj_raw = (
                    batch["cam_traj"]
                    .to(weight_dtype)
                    .to(accelerator.device, non_blocking=True)
                )
                memorized_pixel_values = (
                    batch["memorized_pixel_values"]
                    .to(weight_dtype)
                    .to(accelerator.device, non_blocking=True)
                )
                camera_traj = (
                    torch.zeros(camera_traj_raw.shape[0], num_frames, 3, 4)
                    .to(weight_dtype)
                    .to(accelerator.device, non_blocking=True)
                )
                plucker_embedding = (
                    torch.zeros(
                        camera_traj_raw.shape[0],
                        num_frames,
                        6,
                        args.height // 8,
                        args.width // 8,
                    )
                    .to(weight_dtype)
                    .to(accelerator.device, non_blocking=True)
                )
                for i in range(camera_traj_raw.shape[0]):
                    camera_traj[i] = xyz_euler_to_three_by_four_matrix_batch(
                        camera_traj_raw[i], relative=True
                    )  # Step, 3, 4
                    plucker_embedding[i] = ray_c2w_to_plucker(
                        rays, camera_traj[i]
                    )  # Step, 6, 72, 128
                conditional_pixel_values = torch.cat(
                    (pixel_values[:, 0:1, :, :, :], memorized_pixel_values), dim=1
                )  # [1, 1+25, 3, 576, 1024]

                latents = tensor_to_vae_latent(pixel_values, vae)  # [1, 25, 4, 72, 128]

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                h_latent = latents.shape[-2]
                w_latent = latents.shape[-1]

                cond_sigmas = rand_log_normal(
                    shape=[
                        bsz,
                    ],
                    loc=-3.0,
                    scale=0.5,
                ).to(latents)
                noise_aug_strength = cond_sigmas[
                    0
                ] 
                cond_sigmas = cond_sigmas[:, None, None, None, None]
                conditional_pixel_values = (
                    torch.randn_like(conditional_pixel_values) * cond_sigmas
                    + conditional_pixel_values
                )
                conditional_latents = tensor_to_vae_latent(
                    conditional_pixel_values, vae
                )  # [1, 1+25, 4, 72, 128]
                conditional_latents = conditional_latents / vae.config.scaling_factor
                conditional_latents_first_frame = conditional_latents[:, 0:1].repeat(
                    1, num_frames, 1, 1, 1
                )
                conditional_latents = torch.cat(
                    (conditional_latents_first_frame, conditional_latents[:, 1:]), dim=2
                )  # [1, 25, 4+4, 72, 128]

                # Sample a random timestep for each image
                # P_mean=0.7 P_std=1.6
                sigmas = rand_log_normal(
                    shape=[
                        bsz,
                    ],
                    loc=0.7,
                    scale=1.6,
                ).to(latents.device)
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                sigmas = sigmas[:, None, None, None, None]

                noisy_latents = latents + noise * sigmas

                timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas]).to(
                    accelerator.device
                )

                inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

                # Get the text embedding for conditioning.
                encoder_hidden_states = encode_image(
                    pixel_values[:, 0, :, :, :].float()
                )

                # Here I input a fixed numerical value for 'motion_bucket_id', which is not reasonable.
                # However, I am unable to fully align with the calculation method of the motion score,
                # so I adopted this approach. The same applies to the 'fps' (frames per second).
                added_time_ids = _get_add_time_ids(
                    7,  # fixed
                    # fps, # fixed
                    # 6, # fixed
                    127,  # motion_bucket_id = 127, fixed
                    noise_aug_strength,  # noise_aug_strength == cond_sigmas
                    encoder_hidden_states.dtype,
                    bsz,
                )
                added_time_ids = added_time_ids.to(latents.device)

                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(
                        bsz, device=latents.device, generator=generator
                    )

                    # Sample masks for the original images.
                    image_mask = random_p < args.conditioning_dropout_prob
                    image_mask = image_mask.repeat(bsz, 1, 1)

                    null_conditioning = torch.zeros_like(encoder_hidden_states)
                    encoder_hidden_states = torch.where(
                        image_mask,
                        null_conditioning.unsqueeze(1),
                        encoder_hidden_states.unsqueeze(1),
                    )

                    # Sample masks for the original images.
                    image_mask_dtype = conditional_latents.dtype
                    image_mask = 1 - (random_p < args.conditioning_dropout_prob).to(
                        image_mask_dtype
                    )
                    image_mask = image_mask.repeat(
                        bsz, conditional_latents.shape[1], 1, 1, 1
                    )
                    mem_mask = 1 - (random_p < 2 * args.conditioning_dropout_prob).to(
                        image_mask_dtype
                    )
                    mem_mask = mem_mask.repeat(bsz, conditional_latents.shape[1], 1, 1, 1)
                    # Final image conditioning.
                    conditional_latents[:, :, :4] = (image_mask * conditional_latents[:, :, :4])
                    conditional_latents[:, :, 4:] = (mem_mask * conditional_latents[:, :, 4:])

                # Concatenate the `conditional_latents` with the `noisy_latents`.
                inp_noisy_latents = torch.cat(
                    [inp_noisy_latents, conditional_latents], dim=2
                )  # [1, 25, 4+8, 72, 128]

                if args.add_plucker and not args.denoise_plucker:
                    inp_noisy_latents = torch.cat(
                        [inp_noisy_latents, plucker_embedding], dim=2
                    )  # [1, 25, 4+8+6, 72, 128]

                # check https://arxiv.org/abs/2206.00364(the EDM-framework) for more details.
                target = latents

                model_pred = unet(
                    inp_noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    added_time_ids=added_time_ids,
                ).sample

                # Denoise the latents
                c_out = -sigmas / ((sigmas**2 + 1) ** 0.5)
                c_skip = 1 / (sigmas**2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents

                weighing = (1 + sigmas**2) * (sigmas**-2.0)

                # MSE loss
                loss = torch.mean(
                    (
                        weighing.float()
                        * (denoised_latents.float() - target.float()) ** 2
                    ).reshape(target.shape[0], -1),
                    dim=1,
                )
                loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(args.per_gpu_batch_size)
                ).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {"train_loss": train_loss, "lr": lr_scheduler.get_lr()[0]},
                    step=global_step,
                )
                train_loss = 0.0

                if accelerator.is_main_process:
                    # save checkpoints!
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)
                    # sample images!
                    if (
                        (global_step % args.validation_steps == 0) or (global_step == 1)
                    ) and accelerator.is_main_process:
                        logger.info(
                            f"Running validation... \n Generating {args.num_validation_images} videos."
                        )
                        logger.info(f"Running validation on GPU: {accelerator.device}")
                        # create pipeline
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        # The models need unwrapping because for compatibility in distributed training mode.
                        pipeline = StableVideoDiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            image_encoder=accelerator.unwrap_model(image_encoder),
                            vae=accelerator.unwrap_model(vae),
                            revision=args.revision,
                            torch_dtype=weight_dtype,
                        )
                        pipeline = pipeline.to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)

                        # run inference
                        val_save_dir = os.path.join(
                            args.output_dir, "validation_images"
                        )

                        if not os.path.exists(val_save_dir):
                            os.makedirs(val_save_dir)

                        with torch.autocast(
                            str(accelerator.device).replace(":0", ""),
                            enabled=accelerator.mixed_precision == "fp16",
                        ):
                            for val_step, batch in enumerate(val_loader):
                                if val_step >= args.num_validation_images:
                                    break
                                # for val_img_idx in (range(args.num_validation_images) if global_step >= 18000 else range(1)):
                                num_frames = args.num_frames
                                images = batch["pixel_values"]
                                first_frame = images[:, 0, :, :, :]
                                camera_traj_raw = batch["cam_traj"].to(
                                    accelerator.device, non_blocking=True
                                )
                                memorized_pixel_values = batch["memorized_pixel_values"]

                                camera_traj = (
                                    torch.zeros(
                                        camera_traj_raw.shape[0], num_frames, 3, 4
                                    )
                                    .to(weight_dtype)
                                    .to(accelerator.device, non_blocking=True)
                                )
                                plucker_embedding = (
                                    torch.zeros(
                                        camera_traj_raw.shape[0],
                                        num_frames,
                                        6,
                                        args.height // 8,
                                        args.width // 8,
                                    )
                                    .to(weight_dtype)
                                    .to(accelerator.device, non_blocking=True)
                                )
                                for i in range(camera_traj_raw.shape[0]):
                                    camera_traj[i] = (
                                        xyz_euler_to_three_by_four_matrix_batch(
                                            camera_traj_raw[i], relative=True
                                        )
                                    )  # Step, 3, 4
                                    plucker_embedding[i] = ray_c2w_to_plucker(
                                        rays, camera_traj[i]
                                    )  # Step, 6, 72, 128

                                video_frames = pipeline(
                                    first_frame,
                                    height=args.height,
                                    width=args.width,
                                    num_frames=num_frames,
                                    decode_chunk_size=8,
                                    motion_bucket_id=127,
                                    fps=7,
                                    noise_aug_strength=0.02,
                                    plucker_embedding=plucker_embedding,
                                    memorized_pixel_values=memorized_pixel_values,
                                ).frames[0]

                                out_file = os.path.join(
                                    val_save_dir,
                                    f"step_{global_step}_val_img_{val_step}.mp4",
                                )
                                concatenated_frames = []
                                for i in range(num_frames):
                                    gt_frame = np.array(images[0, i]).transpose(1, 2, 0)
                                    gt_frame = ((gt_frame / 2 + 0.5) * 255).astype(
                                        np.uint8
                                    )
                                    pred_frame = np.array(video_frames[i])
                                    concatenated_frame = np.concatenate(
                                        (gt_frame, pred_frame), axis=0
                                    )
                                    concatenated_frames.append(concatenated_frame)
                                export_to_gif(concatenated_frames, out_file, 7)

                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())

                        del pipeline
                        torch.cuda.empty_cache()

                # save checkpoints!
                # DeepSpeed Model and Optimizer will take a long time if we do it in main process
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            image_encoder=accelerator.unwrap_model(image_encoder),
            vae=accelerator.unwrap_model(vae),
            unet=unet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )
    accelerator.end_training()


if __name__ == "__main__":
    main()
