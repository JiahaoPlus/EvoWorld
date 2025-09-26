#!/bin/bash

# Activate the Python environment
source /scratch/ayuille1/jwang384/miniconda3/bin/activate evoworld
echo $CONDA_DEFAULT_ENV
echo $CONFIG_NAME

# Base directory containing episode folders.
BASE_DIR="data/Segment_Loop/train"

# Loop over every directory matching the pattern "episode_*".
for EPISODE_PATH in "$BASE_DIR"/*; do
    
    # Make sure it's actually a directory (and not matching an empty glob, etc.).
    if [[ -d "$EPISODE_PATH" ]]; then
        echo "Processing: $EPISODE_PATH"
        
        # Build paths for source, target, etc.
        SOURCE_DIR="$EPISODE_PATH/panorama"
        TARGET_DIR="$EPISODE_PATH/perspective_look_at_center"
        CAMERA_FILE="$EPISODE_PATH/camera_poses.txt"
        OUTPUT_CAMERA_FILE="$EPISODE_PATH/camera_poses_look_at_center.txt"
        
        # Run the python script for this episode.
        python -m evoworld.reprojection.pano_to_pers \
            --data_folder "$SOURCE_DIR" \
            --output_folder "$TARGET_DIR" \
            --camera_file "$CAMERA_FILE" \
            --output_camera_file "$OUTPUT_CAMERA_FILE"
        
        echo "Finished: $EPISODE_PATH"
        echo "----------------------------------------"
    fi
done

echo "All done!"
