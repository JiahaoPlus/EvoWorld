#!/bin/bash
nvidia-smi
# Activate the Python environment
source /scratch/ayuille1/jwang384/miniconda3/bin/activate evoworld

# configuration file, you can add more config files in the config folder
CONFIG_NAME="deepspeed_o1_4gpu"

# global seed
SEED=42

# data settings
DATASET_NAME="Curve_Loop"
WIDTH=1024
HEIGHT=576
NUM_FRAMES=25

# model & trainer settings
PRETRAIN_MODEL="MODELS/stable-video-diffusion-img2vid-xt-1-1"
STEP=6
SAVE_INTERVAL=5000
GRAD_ACCUM_STEP=4
LR="1e-5"
LR_WARMUP_STEP=500
LR_SCHEDULER="cosine"
PRECISION="fp16"
VALIDATION_STEP=5
NUM_VALIDATION_IMAGES=3
RESUME_FROM="latest"
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
BATCH_SIZE_PER_GPU=1
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
WORLD_SIZE=$((GPUS_PER_NODE * GRAD_ACCUM_STEP * BATCH_SIZE_PER_GPU))

export WANDB_RUN_NAME="data-${DATASET_NAME}-lr-${LR}-step-${STEP}-bs-${BATCH_SIZE_PER_GPU}x${GPUS_PER_NODE}x${GRAD_ACCUM_STEP}-${CURRENT_TIME}"
export WANDB_API_KEY='46e587ae4112a04da96b68ba807395204be787c9'
export WANDB_ENTITY='lucassss'
export WANDB_PROJECT='evoworld'

for var in CONFIG_NAME SEED DATASET_NAME WIDTH HEIGHT NUM_FRAMES PRETRAIN_MODEL STEP SAVE_INTERVAL GRAD_ACCUM_STEP LR LR_WARMUP_STEP LR_SCHEDULER WORLD_SIZE PRECISION VALIDATION_STEP NUM_VALIDATION_IMAGES RESUME_FROM; do
    echo "$var: ${!var}"
done
echo "Runing will be logged to WANDB project: $WANDB_PROJECT, entity: $WANDB_ENTITY, run name: $WANDB_RUN_NAME"

echo "Current Env: $(conda info --envs | grep '*' | awk '{print $1}')"

accelerate launch --config_file="config/${CONFIG_NAME}.yaml" \
    evoworld/trainer/train_evoworld.py \
    --base_folder=data/$DATASET_NAME \
    --pretrained_model_name_or_path=$PRETRAIN_MODEL \
    --num_frames=$NUM_FRAMES \
    --width=$WIDTH \
    --height=$HEIGHT \
    --output_dir="MODELS/$DATASET_NAME-$CONFIG_NAME-lr-$LR-step-$STEP-worldsize-$WORLD_SIZE" \
    --logging_dir="MODELS/$DATASET_NAME-$CONFIG_NAME-lr-$LR-step-$STEP-worldsize-$WORLD_SIZE/logs" \
    --per_gpu_batch_size=$BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps=$GRAD_ACCUM_STEP \
    --gradient_checkpointing \
    --max_train_steps=$STEP \
    --checkpointing_steps=$SAVE_INTERVAL \
    --checkpoints_total_limit=4 \
    --learning_rate=$LR \
    --lr_warmup_steps=$LR_WARMUP_STEP \
    --lr_scheduler=$LR_SCHEDULER \
    --scale_lr \
    --seed=$SEED \
    --mixed_precision=$PRECISION \
    --validation_steps=$VALIDATION_STEP \
    --num_validation_images=$NUM_VALIDATION_IMAGES \
    --report_to=wandb \
    --add_plucker \
    --resume_from_checkpoint=$RESUME_FROM \
    --push_to_hub \
    --hub_model_id CometsFeiyu/evoworld-$DATASET_NAME-lr-$LR-step-$STEP-worldsize-$WORLD_SIZE
