#!/bin/bash

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

CKPT=MODELS/evoworld_curve_unity
OUTPUT_ROOT=output
VIDEO_PATH=$OUTPUT_ROOT/$(basename $CKPT)/eval_unity_curve
NUM_VIDEO=200
RESULT_PATH="eval_score.json"
SEGMENT_ID=2  # Change this to evaluate different segments (0, 1, 2)

python -m evoworld.metrics.calculate_all_metrics \
    --data_path $VIDEO_PATH \
    --gt_subdir "predictions_gt_$SEGMENT_ID" \
    --gen_subdir "predictions_$SEGMENT_ID" \
    --result_file $RESULT_PATH \
    --test_length 25 \
    --num_video $NUM_VIDEO