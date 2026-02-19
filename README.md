# Renal fMRI: Automated Segmentation & Registration Framework

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9-ff69b4.svg" /></a>
<a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/TensorFlow-2.8-2BAF2B.svg" /></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
<a href="https://onlinelibrary.wiley.com/doi/10.1002/mrm.70288"><img src="https://img.shields.io/badge/DOI-10.1002/mrm.70288-blue.svg" /></a>

---

## Publication

This repository contains the official implementation of the methods described in:
"Fully Automated Deep Learning-Driven Postprocessing Pipeline for Multiparametric Renal MRI"

Magnetic Resonance in Medicine (MRM), 2026

DOI: [10.1002/mrm.70288](https://doi.org/10.1002/mrm.70288)

---

## Overview

This framework provides a complete end-to-end pipeline for automated analysis of multiparametric 
renal MRI data. It processes six MRI contrasts (DIXON, T1 mapping, T2 mapping, T2* BOLD, ASL renal 
blood flow, and ADC maps) in under 15 seconds per scan, 
producing co-registered segmentations and clinically relevant quantitative features.

**Key Features:**

- Constrained Contrastive Learning (CCL): Pre-train on unlabeled data to learn tissue-specific representations
- Multi-class Segmentation: 3-class (whole kidney) or 5-class (cortex/medulla) segmentation
- Deformable Registration: Affine + non-rigid registration to align all contrasts to DIXON space
- Feature Extraction: Automated volume measurements and region-specific quantitative analysis
- Fast Inference: Complete pipeline in ~15 seconds per scan on GPU

---

## Pipeline Architecture
<div align="center">

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Pre-training  │───▶│  Segmentation   │───▶│   Registration  │
│  (Contrastive   │    │  Fine-tuning    │    │   Affine +      │
│   Learning)     │    │                 │    │   Non-rigid     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Constraint Maps │    │ Kidney Masks    │    │ Aligned Images  │
│  (T2, T2* PCA   │    │ (3 or 5 class)  │    │ & Warped Masks  │
│   Clustering)   │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```
</div>

---


## Installation

### Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA support (recommended for training)
- 16GB+ RAM

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/lab-midas/Renal-fMRI-Framework.git

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
   
---
   
## Data Preparation

### Directory Structure

Your data must be organized in the following structure after resampling:

```
data/resampled/
├── P_01_A/                          # Patient/Subject ID
│   ├── DIXON/                        # Template contrast
│   │   ├── imagesTr/
│   │   │   └── water_s001.nii
│   │   └── labelsTr/
│   │       ├── P01_A_dixon_left_volume.nii.gz
│   │       ├── P01_A_dixon_right_volume.nii.gz
│   │       ├── P01_A_dixon_left_cortex.nii.gz
│   │       └── P01_A_dixon_right_cortex.nii.gz
│   ├── BOLD/                          # T2* imaging
│   │   ├── imagesTr/
│   │   │   ├── e1_s025.nii
│   │   │   ├── e2_s025.nii
│   │   │   └── e3_s025.nii
│   │   └── labelsTr/
│   │       ├── P01_A_bold_left_volume.nii.gz
│   │       └── P01_A_bold_right_volume.nii.gz
│   ├── T1_mapping_VIBE/                # T1 mapping
│   ├── T1_mapping_fl2d/                 # FLASH T1
│   ├── T2_mapping_PREP/                  # T2 mapping
│   ├── ASL/                              # Arterial spin labeling
│   └── Diffusion/                        # DWI/ADC
├── P_02_A/
└── V_01_A/                           # Healthy volunteer
```

## Supported Contrasts

| Contrast           | Description       | Files                        |
|-------------------|-----------------|------------------------------|
| DIXON             | Water-fat imaging | water, fat, in-phase, out-phase |
| BOLD              | T2* BOLD imaging  | Multiple echoes (e1, e2, e3) |
| T1_mapping_VIBE   | T1 mapping (VIBE) | Multiple flip angles          |
| T1_mapping_fl2d   | T1 mapping (FLASH)| Multiple flip angles          |
| T2_mapping_PREP   | T2 mapping        | t2map, t2prep                |
| ASL               | Renal blood flow  | M0, RBF maps                 |
| Diffusion         | DWI/ADC           | Multiple b-values             |

---

## Step 1: Pre-training with Contrastive Learning

Pre-train the encoder using Constrained Contrastive Learning (CCL) on unlabeled T2 and T2* images.

1. Generate Constraint Maps: First, generate constraint maps that capture tissue patterns:
    ```bash
   python data/pretrain_Data_Generator.py \
    --save_dir=/path/to/constraint_maps \
    --data_dir=/path/to/your/data \
    --contrast=BOLD \
    --num_clusters=20
   
**Arguments:**

- save_dir: Output directory for constraint maps
- data_dir: Input directory with resampled data
- contrast: Contrast to process (BOLD or T2_mapping_PREP)
- num_clusters: Number of K-means clusters (default: 20)
- output_size: Output image size (default: 256)

2. Configure Training

Edit configs/pretrain.py to set your parameters

3. Run Pre-training
    ```bash
    python pretrain.py [--gpu GPU_ID] [--debug] [--resume CHECKPOINT]
   
- --gpu: GPU ID to use (default: "2")
- --debug: Run with limited data for testing
- --resume: Resume from a checkpoint

Output: Pre-trained weights are saved to: checkpoints/pretrain/{contrast}/{experiment_name}/weights_{epochs}.hdf5

These weights will be used to initialize the segmentation network in the next step.

---
## Step 2: Segmentation Fine-tuning
Fine-tune the pre-trained encoder for kidney segmentation.
1. Configure Fine-tuning

Edit configs/finetune.py

2. Run Fine-tuning
    ```bash
    # Basic trainingp
    python finetune.py
    
    # With specific fold and task
    python finetune.py --gpu 0 --fold 1 --task volume
    
    # With pre-trained weights
    python finetune.py --resume /path/to/pretrained/weights.hdf5

Output: Segmentation model weights saved to {base_save_dir}/{experiment_name}/

---
## Step 3: Registration

1. Prepare Data for Registration

Convert resampled NIfTI files to HDF5 format for efficient loading:
```bash
python data.generate_reg_data.py \
    --input_dir /path/to/resampled/data \
    --output_dir /path/to/output/dir \
    --contrasts DIXON BOLD T1_mapping_VIBE ASL Diffusion
```

2. Train Affine Registration
Edit configs/registration_affine.py and run:

```bash
python train_reg_affine.py --gpu 0
```

3. Train Non-rigid Registration
Edit configs/registration_nonrigid.py and run:

```bash
python train_reg_nonrigid.py --gpu 0
```

---
## Inference

### Segmentation Inference

Use the provided Jupyter notebook for segmentation inference:

```bash
jupyter notebook scripts/inference_segmentation.ipynb
```

This notebook performs segmentation on contrast images included during training, 
offering two output modes: full-volume segmentation or cortex–medulla segmentation.

### Registration Inference

Register all contrasts to DIXON and propagate masks:

```bash
jupyter notebook scripts/inference_registration.ipynb
```
This notebook will:

- Load pre-trained affine and non-rigid models
- Register all contrasts to DIXON space
- Propagate DIXON segmentation masks to all contrasts
- Save registered images, warped masks, and flow fields

---
## Citation
If you use this code in your research, please cite:
```bash
@article{ghoulautomated,
  title={Automated Coregistered Segmentation for Volumetric Analysis of Multiparametric Renal MRI},
  author={Ghoul, Aya and Liang, Cecilia and Loster, Isabelle and Umapathy, Lavanya and Kühn, 
  Bernd and Martirosian, Petros and Seith, Ferdinand and Gatidis, Sergios and Küstner, Thomas},
  journal={Magnetic resonance in medicine}
}
```

---
## Acknowledgements

This project was developed with the help of the following repository:

- [Constrained Contrastive Learning](https://github.com/lunastra26/multi-contrast-contrastive-learning) – for pre-training contrastive learning technique.  
- [VoxelMorph](https://github.com/voxelmorph/voxelmorph) – for the motion estimation losses and metrics.

---
## License

Distributed under the MIT License. See LICENSE for more information.

---

## Contact

For questions or issues, please open a GitHub issue or contact the corresponding 
author at: [aya.ghoul@med.uni-tuebingen.de](mailto:aya.ghoul@med.uni-tuebingen.de)
