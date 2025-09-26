#!/bin/bash

SCRIPT_PATH="evoworld/reprojection/reproject_vggt_open3d_for_train.sh"

# Define fixed parameters
DATA_FOLDER="data/Segment_Loop"   # <-- Replace with actual data folder
# SEGMENT_ID=1      # <-- Replace with actual segment ID
CHUNK_NUM=16      # Total number of chunks

for CHUNK_ID in $(seq 0 $((CHUNK_NUM-1))); do
    echo "Launching chunk $CHUNK_ID..."
    bash "$SCRIPT_PATH" "$DATA_FOLDER" "$SEGMENT_ID" "$CHUNK_ID" "$CHUNK_NUM" &
done

# Wait for all background jobs to finish (only if using & above)
wait
echo "All chunks completed."