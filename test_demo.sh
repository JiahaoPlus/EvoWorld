#!/bin/bash

W=1024
H=576
CKPT=MODELS/trained_deepspeed_o1_add_cam_plucker_add_mem_on_Curve_Loop_30000_3d_vggt_open3d_camera_aligned
DATA=data/Curve_Loop/test/episode_0004

# Activate the Python environment
source /scratch/ayuille1/jwang384/miniconda3/bin/activate evoworld

python -m evoworld.inference.forward_evoworld \
    --ckpt $CKPT \
    --data $DATA \
    --width $W \
    --height $H \
    --verbose \
    --reprojection_name rendered_panorama_vggt_open3d \
    --output_name demo

echo "Finished."
