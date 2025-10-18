
# EvoWorld

**Official implementation of**  
**[EvoWorld: Evolving Panoramic World Generation with Explicit 3D Memory](https://arxiv.org/abs/2510.01183)**  

EvoWorld is a generative world model that explicitly reconstructs and utilizes a 3D memory for egocentric video generation. Starting from a single panoramic image and a sequence of camera poses, EvoWorld synthesizes future frames with strong 3D consistency by projecting an evolving 3D point cloud onto future views as conditioning for a video diffusion model.

<p align="center">
  <img src="assets/overview.png" alt="EvoWorld Pipeline Overview" width="70%">
</p>

## � Installation

```bash
git clone git@github.com:JiahaoPlus/EvoWorld.git --recursive
cd evoworld
conda create -n evoworld python=3.11
conda activate evoworld
pip install -r requirements.txt
conda install -c conda-forge "cudnn>=9,<10"
```

## 📦 Released Models & Datasets

We have released:

- **Model weights trained on Unity Curve Path (subset):**
  [Evoworld_Unity_Curve_Path (Hugging Face)](https://huggingface.co/CometsFeiyu/Evoworld_Unity_Curve_Path)
- **Corresponding dataset:**
  [Google Drive link](https://drive.google.com/file/d/1xkVi83huO7WkRm6_XZ9AXJEhjvVmux0q/view?usp=drive_link)


### How to use

**1. Download model weights and dataset**
  - Download model weights from Hugging Face and place in the `MODELS` directory.
  - Download the dataset from Google Drive.
  - **After downloading the dataset, extract it:**
    ```bash
    tar xvf path/to/dataset.tar
    ```
  - Update the model path in `run_single_segment.sh` and `run_unified_pipeline.sh` to point to your downloaded weights.

**2. Run model on provided example**
   - For **single clip generation**, run:
     ```bash
     bash run_single_segment.sh
     ```
   - For **3-clip iterative generation**, run:
     ```bash
     bash run_unified_pipeline.sh
     ```
   - Make sure the model path in these scripts matches your downloaded weights.

**3. Run model across the whole test set**
   - Change your model path in `inference_unity_curve.sh` to the downloaded weights.
   - Execute:
     ```bash
     bash inference_unity_curve.sh
     ```

**4. Get evaluation metrics**
   - Run:
     ```bash
     bash calculate_metrics.sh
     ```

```bash
git clone git@github.com:JiahaoPlus/EvoWorld.git --recursive
cd evoworld
conda create -n evoworld python=3.11
conda activate evoworld
pip install -r requirements.txt
conda install -c conda-forge "cudnn>=9,<10"
```

## 🎬 Demo

### Single-clip demo
```bash
bash run_single_segment.sh
```

### Long-horizon loop demo
```bash
bash run_unified_pipeline.sh
```

## 🗺️ Spatial360 Dataset

Comming soon

<!-- ## 🗺️ Spatial360 Dataset

We release **Spatial360**, the first dataset for long-range and loop-closure exploration with 360° panoramic videos and poses. It spans:
- Synthetic outdoor: Unity and UE5
- Indoor: Habitat (HM3D and Matterport3D)
- Real-world: Captured using Insta360 -->

<!-- ## 📦 Dataset Preparation

### Download Panoramic Videos and Camera Poses
You can download pre-processed clips from the [Spatial360](https://github.com/todo) dataset:
```bash
gdown https://drive.google.com/uc?id=VIDEO_FILE_ID
```

### Download Pre-generated 3D Reprojections (Optional)
```bash
gdown https://drive.google.com/uc?id=REPROJECTION_FILE_ID
``` -->

<!-- ### Or Generate Reprojections for Training
#### Step 1: Convert panoramic images to perspective views
```bash
bash scripts/reprojection/pano_to_pers_for_train.sh
```

#### Step 2: Reconstruct 3D memory and project to target views
This step uses VGGT [CVPR 2025] to build colored point clouds and Open3D for rendering.
```bash
bash reproject_vggt_open3d_for_train_sbatch.sh
``` -->

## 🏋️ Training
```bash
# Train EvoWorld
bash train.sh
```

## 🧪 Inference

### 1. Single-clip generation
```bash
sbatch scripts/inference_single/test_evoworld_batch.sh
```

### 2. Long-horizon loop consistency generation
```bash
sbatch scripts/inference_multiple/loop_consistency_evoworld.sh
```

## 📎 Acknowledgements

This codebase builds upon:
- [VGGT](https://github.com/jianyuann/vggt)
- [360-1M](https://github.com/MattWallingford/360-1M)
- [SVD_Xtend](https://github.com/pixeli99/SVD_Xtend)
- [GenEx](https://github.com/GenEx-world/genex)