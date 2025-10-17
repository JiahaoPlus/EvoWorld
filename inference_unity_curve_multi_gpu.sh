#!/bin/bash

# Unified Loop Consistency Pipeline Runner
# This script runs the unified pipeline that combines generation and reconstruction

# link to correct cudnn and cuda
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

CKPT=MODELS/evoworld_curve_unity
BASE_FOLDER=data/Curve_Loop
OUTPUT_ROOT=output
SAVE_DIR=$OUTPUT_ROOT/$(basename $CKPT)/eval_unity_curve
START_IDX=0
NUM_DATA_PER_GPU=50 # Adjust this based on your NUM_GPUS: total data = NUM_GPUS * NUM_DATA_PER_GPU
NUM_SEGMENTS=3
CURVE_PATH=true

# Number of GPUs to use. If not set externally, try to detect with nvidia-smi.
# You can override by exporting NUM_GPUS before running this script.
if [ -z "$NUM_GPUS" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
        # if nvidia-smi returned nothing or 0, fallback to 1
        if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -le 0 ]; then
            NUM_GPUS=1
        fi
    else
        NUM_GPUS=1
    fi
fi

echo "Using NUM_GPUS=$NUM_GPUS"

echo "Running unified loop consistency pipeline..."
echo "Checkpoint: $CKPT"
echo "Base folder: $BASE_FOLDER"
echo "Save dir: $SAVE_DIR"
echo "Number of segments: $NUM_SEGMENTS"

PIDS=()
for (( GPU=0; GPU<NUM_GPUS; GPU++ )); do
    # compute start idx for this GPU
    PROC_START_IDX=$(( START_IDX + GPU * NUM_DATA_PER_GPU ))


    # build per-process command
    CMD_GPU="python unified_loop_consistency.py \
        --unet_path $CKPT \
        --svd_path $CKPT \
        --base_folder $BASE_FOLDER \
        --save_dir $SAVE_DIR \
        --num_data $NUM_DATA_PER_GPU \
        --start_idx $PROC_START_IDX \
        --num_segments $NUM_SEGMENTS \
        --num_frames 25 \
        --save_frames"

    if [ "$CURVE_PATH" = true ]; then
        CMD_GPU="$CMD_GPU --curve_path"
    fi

    echo "Launching GPU $GPU -> start_idx=$PROC_START_IDX, save_dir=$SAVE_DIR"
    echo "Command: CUDA_VISIBLE_DEVICES=$GPU $CMD_GPU"

    # Run the process bound to this GPU in background and capture its PID
    CUDA_VISIBLE_DEVICES=$GPU bash -c "$CMD_GPU" > "$SAVE_DIR/run_gpu${GPU}.log" 2>&1 &
    PIDS+=("$!")
done

echo "Launched ${#PIDS[@]} processes, waiting for them to finish..."

# wait for all background processes
for pid in "${PIDS[@]}"; do
    wait "$pid" || echo "Process $pid exited with non-zero status"
done

echo "Unified pipeline completed for all GPUs!"



