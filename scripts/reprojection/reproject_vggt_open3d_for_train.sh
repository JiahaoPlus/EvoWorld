#!/bin/bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

DATA_FOLDER=$1
CHUNK_ID=$2
CHUNK_NUM=$3

SUB=perspective_look_at_center
CAM=camera_poses_look_at_center.txt
OUTPUT_SUBDIR=rendered_panorama_vggt_open3d_camera_aligned_new_code

python -m evoworld.reprojection.reproject_vggt_open3d \
    --data_folder $DATA_FOLDER \
    --image_subdir $SUB \
    --no_mask_sky \
    --prediction_mode depth_unproject \
    --camera_file $CAM \
    --output_subdir $OUTPUT_SUBDIR \
    --chunk_num $CHUNK_NUM \
    --chunk_id $CHUNK_ID \
    --only_render_last_24_frame