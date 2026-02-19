# Data Preprocessing for Renal MRI Framework

This directory contains scripts for preprocessing multi-parametric renal MRI data and generating constraint maps for contrastive learning.

## 📋 Scripts

| Script                       | Purpose |
|------------------------------|---------|
| `generate_ccl_maps.py`       | Generate constraint maps from multi-contrast images |

## 📁 Required Data Structure

Place your resampled NIfTI files in the following structure:

```
data/resampled/
├── P_01_A/                          # Patient/Subject ID
│   ├── BOLD/                         # Contrast type
│   │   ├── imagesTr/                  # Input images
│   │   │   └── e1_s025.nii
│   │   └── labelsTr/                  # Segmentation masks
│   │       ├── P01_A_bold_left_cortex.nii.gz
│   │       ├── P01_A_bold_left_volume.nii.gz
│   │       ├── P01_A_bold_right_cortex.nii.gz
│   │       └── P01_A_bold_right_volume.nii.gz
│   ├── T2_mapping_PREP/
│   │   ├── imagesTr/
│   │   │   └── t2prep_s028.nii
│   │   └── labelsTr/
│   │       ├── P01_A_t2_left_cortex.nii.gz
│   │       ├── P01_A_t2_left_volume.nii.gz
│   │       ├── P01_A_t2_right_cortex.nii.gz
│   │       └── P01_A_t2_right_volume.nii.gz
│   ├── T1_mapping_VIBE/
│   ├── ASL/
│   ├── Diffusion/
│   └── DIXON/
├── P_02_A/
│   └── ...
└── V_01_A/
    └── ...
```

## 🚀 Usage

### Generate Constraint Maps

From the project root directory:

```bash
python data/generate_ccl_maps.py \
    --save_dir=/path/to/output/Constraint_maps/param \
    --data_dir=/path/to/data/resampled \
    --contrast=BOLD \
    --num_clusters=20 \
    --output_size=256
```

### Batch Processing for Multiple Contrasts

```bash
#!/bin/bash
# save as `run_preprocessing.sh`

CONTRASTS=("BOLD" "T2_mapping_PREP" "T1_mapping_VIBE")
DATA_DIR="/path/to/data/resampled"
SAVE_DIR="/path/to/output/Constraint_maps/param"

for contrast in "${CONTRASTS[@]}"; do
    echo "Processing $contrast..."
    python data/pretrain_Data_Generator.py \
        --save_dir=$SAVE_DIR \
        --data_dir=$DATA_DIR \
        --contrast=$contrast \
        --num_clusters=20
done
```

## 📊 Output Format

Generated constraint maps are saved as MATLAB `.mat` files:

```python
import scipy.io as sio
import numpy as np

# Load constraint map
data = sio.loadmat('Constraint_map_P_01_A_20.mat')

# Access components
constraint_map = data['param']  # Shape: (H, W, D)
original_image = data['img']     # Shape: (H, W, D, C)
segmentation_mask = data['mask'] # Shape: (H, W, D)

print(f"Unique clusters: {np.unique(constraint_map)}")
```

## 🔧 Configuration

### Key Parameters

| Parameter | Recommended Value | Description |
|-----------|-------------------|-------------|
| `num_clusters` | 20-30 | Higher values capture more tissue subtypes |
| `num_pc_components` | 4-6 | Trade-off between noise reduction and detail |
| `output_size` | 256 | Must match network input size |


## 🐛 Common Issues

### "No module named 'utils'"
```bash
# Run from project root
cd /path/to/Renal-fMRI-Framework
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### "FileNotFoundError: No subjects found"
```bash
# Verify directory structure
ls -la /path/to/data/resampled/P_01_A/BOLD/imagesTr/
```

### "KeyError: 'param' in output file"
```bash
# Check if file was properly saved
file /path/to/output/Constraint_maps/param/BOLD/Constraint_map_P_01_A_20.mat
```

## 📚 References

This preprocessing follows the method described in:
> [Paper Title], Magnetic Resonance in Medicine, 2026
> DOI: 10.1002/mrm.70288

For the complete pipeline, see the main [README.md](../README.md).