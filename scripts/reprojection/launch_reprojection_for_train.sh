#!/bin/bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
SCRIPT_PATH="scripts/reprojection/reproject_vggt_open3d_for_train.sh"

# Define fixed parameters
DATA_FOLDER="data/unity_curve/train"   # <-- Replace with actual data folder
CHUNK_NUM=16      # Total number of chunks

for CHUNK_ID in $(seq 0 $((CHUNK_NUM-1))); do
    echo "Launching chunk $CHUNK_ID..."
    bash "$SCRIPT_PATH" "$DATA_FOLDER" "$CHUNK_ID" "$CHUNK_NUM" 
done

# Wait for all background jobs to finish (only if using & above)
wait
echo "All chunks completed."