"""
Configuration file for affine registration pre-training.
All hyperparameters and settings are centralized here.
"""

import os
from pathlib import Path

# =============================================================================
# GPU SETTINGS
# =============================================================================
# Specify which GPUs to use (comma-separated for multiple)
gpus_available = '3'

# =============================================================================
# DATA PARAMETERS
# =============================================================================
# Image dimensions
img_size_x = 256
img_size_y = 256

# Dataset settings
dataset = 'renfi'  # Dataset name
num_channels = 1  # Number of image channels

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
batch_size = 8  # Batch size for training
lr_pretrain = 1e-4  # Learning rate for registration
initial_epoch = 0  # Starting epoch (for resuming training)
num_epochs = 500  # Total number of training epochs
num_classes = 1  # Number of classes for segmentation masks

# =============================================================================
# LOSS PARAMETERS
# =============================================================================
ft_training = True  # Fine-tuning mode

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
# Get user's home directory for default paths
home_dir = "/home/dir"  # todo: Update with your home directory

# Data directory containing groupwise HDF5 files
data_dir = f'{home_dir}/Renal_fMRI/data/groupwise_all' # todo

# Subject split files (from cross-validation)
val_sub = f'{home_dir}/Renal_fMRI/data/folds/test_f1.npy'  # Validation subjects
train_sub = f'{home_dir}/Renal_fMRI/data/folds/train_f1.npy'  # Training subjects

# Output labels flag
out_labels = True

# Optional: Path to pre-trained checkpoint for resuming training
checkpoint_path = f'{home_dir}/Renal_fMRI/logs_paper/registration_to_dixon/groupwise/affine_PCA_sm_l1_2d_weighted/checkpoints/model_epoch_0500.h5'

# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================
# Template contrast (fixed image)
contrast_template = ['DIXON']

# Moving contrasts (images to be registered to template)
contrast_moving = ['DIXON', 'T1_mapping_VIBE', 'T1_mapping_fl2d', 'BOLD', 'ASL', 'Diffusion']

# Experiment name
experiment_name = "affine_all_selected"

# Feature extraction options
out_features = False
clip = False
weighted = True
pca_template = False
affine = True
num_contrasts = len(contrast_moving)

# Mask type for region weighting
type_mask = 'bbox'  # 'bbox' for bounding box, 'distance' for distance transform

# Base directory for saving results
base_save_dir = f'{home_dir}/Renal_fMRI/logs_paper/test'

# =============================================================================
# WANDB (WEIGHTS & BIASES) CONFIGURATION
# =============================================================================
# Set your WandB API key and entity for experiment tracking
wandb_key = None  # Replace with your key
wandb_entity = None  # Replace with your username

# =============================================================================
# DERIVED PATHS (automatically created from base paths)
# =============================================================================
# Full save directory including experiment name
full_save_dir = os.path.join(base_save_dir, 'groupwise', experiment_name)


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================
def validate_config():
    """Validate configuration parameters."""

    # Check image dimensions
    assert img_size_x > 0 and img_size_y > 0, "Image dimensions must be positive"

    # Check batch size
    assert batch_size > 0, "Batch size must be positive"

    # Check learning rate
    assert lr_pretrain > 0, "Learning rate must be positive"

    # Check number of epochs
    assert num_epochs > 0, "Number of epochs must be positive"

    # Check contrasts
    assert len(contrast_template) > 0, "At least one template contrast must be specified"
    assert len(contrast_moving) > 0, "At least one moving contrast must be specified"

    print("✓ Configuration validation passed")
    return True


# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================
def print_config_summary():
    """Print a formatted summary of the configuration."""

    print("\n" + "=" * 60)
    print("AFFINE REGISTRATION CONFIGURATION")
    print("=" * 60)

    print("\nDATA PARAMETERS:")
    print(f"  • Image size: {img_size_x} x {img_size_y}")
    print(f"  • Dataset: {dataset}")
    print(f"  • Channels: {num_channels}")

    print("\nCONTRASTS:")
    print(f"  • Template (fixed): {contrast_template}")
    print(f"  • Moving: {contrast_moving}")
    print(f"  • Number of moving contrasts: {num_contrasts}")

    print("\nTRAINING PARAMETERS:")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Learning rate: {lr_pretrain}")
    print(f"  • Epochs: {num_epochs}")

    print("\nREGISTRATION OPTIONS:")
    print(f"  • Weighted: {weighted}")
    print(f"  • Affine: {affine}")
    print(f"  • Mask type: {type_mask}")
    print(f"  • PCA template: {pca_template}")

    print("\nPATHS:")
    print(f"  • Data directory: {data_dir}")
    print(f"  • Save directory: {full_save_dir}")
    print(f"  • Training subjects: {train_sub}")
    print(f"  • Validation subjects: {val_sub}")

    print("\nEXPERIMENT:")
    print(f"  • Experiment name: {experiment_name}")

    print("\n" + "=" * 60 + "\n")


# =============================================================================
# CONVENIENCE FUNCTIONS FOR DIFFERENT CONFIGURATIONS
# =============================================================================

def get_config_for_template(template_name: str, custom_params: dict = None):
    """
    Get configuration for a specific template contrast.

    Args:
        template_name: Template contrast name
        custom_params: Dictionary of parameters to override

    Returns:
        Modified configuration dictionary
    """
    # Create a copy of the current configuration
    import copy
    config = copy.deepcopy(globals())

    # Update template
    config['contrast_template'] = [template_name]
    config['experiment_name'] = f"affine_template_{template_name}"
    config['full_save_dir'] = os.path.join(config['base_save_dir'], 'groupwise', config['experiment_name'])

    # Apply custom parameters if provided
    if custom_params:
        for key, value in custom_params.items():
            if key in config:
                config[key] = value

    return config


def get_config_without_weighting(custom_params: dict = None):
    """Get configuration without mask weighting."""
    config = get_config_for_template(contrast_template[0], custom_params)
    config['weighted'] = False
    config['experiment_name'] = config['experiment_name'].replace('weighted', 'unweighted')
    config['full_save_dir'] = os.path.join(config['base_save_dir'], 'groupwise', config['experiment_name'])
    return config


# =============================================================================
# PREDEFINED CONFIGURATIONS
# =============================================================================

# Configuration for DIXON template
DIXON_TEMPLATE_CONFIG = {
    'contrast_template': ['DIXON'],
    'contrast_moving': ['DIXON', 'T1_mapping_VIBE', 'T1_mapping_fl2d', 'BOLD', 'ASL', 'Diffusion'],
    'batch_size': 8,
    'num_epochs': 500,
    'weighted': True,
    'experiment_name': 'affine_dixon_template_weighted'
}

# Configuration for BOLD template
BOLD_TEMPLATE_CONFIG = {
    'contrast_template': ['BOLD'],
    'contrast_moving': ['DIXON', 'T1_mapping_VIBE', 'T1_mapping_fl2d', 'ASL', 'Diffusion'],
    'batch_size': 8,
    'num_epochs': 500,
    'weighted': True,
    'experiment_name': 'affine_bold_template_weighted'
}

# Configuration without weighting
UNWEIGHTED_CONFIG = {
    'weighted': False,
    'experiment_name': 'affine_all_selected_unweighted'
}

# =============================================================================
# RUN VALIDATION ON IMPORT
# =============================================================================
# Validate configuration when this file is imported
if __name__ != "__main__":
    validate_config()
    print_config_summary()