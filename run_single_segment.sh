#!/bin/bash

# Unified Loop Consistency Pipeline Runner
# This script runs the unified pipeline that combines generation and reconstruction

CKPT=MODELS/trained_deepspeed_o1_add_cam_plucker_add_mem_on_Curve_Loop_30000_3d_vggt_open3d_camera_aligned
BASE_FOLDER=data/Curve_Loop/test/episode_0004
OUTPUT_ROOT=output
SAVE_DIR=$OUTPUT_ROOT/$(basename $CKPT)/unified_single_demo
START_IDX=0
NUM_DATA_PER_GPU=1
NUM_SEGMENTS=3
CURVE_PATH=true

source /scratch/ayuille1/jwang384/miniconda3/bin/activate evoworld

echo "Running unified loop consistency pipeline..."
echo "Checkpoint: $CKPT"
echo "Base folder: $BASE_FOLDER"
echo "Save dir: $SAVE_DIR"
echo "Number of segments: $NUM_SEGMENTS"

CMD="python unified_loop_consistency.py \
    --unet_path $CKPT \
    --svd_path $CKPT \
    --base_folder $BASE_FOLDER \
    --save_dir $SAVE_DIR \
    --num_data $NUM_DATA_PER_GPU \
    --start_idx $START_IDX \
    --num_segments $NUM_SEGMENTS \
    --num_frames 25 \
    --save_frames \
    --single_segment"

if [ "$CURVE_PATH" = true ]; then
    CMD="$CMD --curve_path"
fi

echo "Command: $CMD"
eval $CMD

echo "Unified pipeline completed!"
