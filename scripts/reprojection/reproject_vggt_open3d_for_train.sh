#!/bin/bash

DATA_FOLDER=$1
SEGMENT_ID=$2
CHUNK_ID=$3
CHUNK_NUM=$4

SUB=perspective_look_at_center
CAM=camera_poses_look_at_center.txt
OUTPUT_SUBDIR=rendered_panorama_vggt_open3d_camera_aligned

# Activate the Python environment
source /scratch/ayuille1/jwang384/miniconda3/bin/activate evoworld

python demo_slurm_open3d_with_camera.py \
    --data_folder $DATA_FOLDER \
    --image_subdir $SUB \
    --no_mask_sky \
    --prediction_mode depth_unproject \
    --camera_file $CAM \
    --output_subdir $OUTPUT_SUBDIR \
    --chunk_num $CHUNK_NUM \
    --chunk_id $CHUNK_ID \
    --only_render_last_24_frame