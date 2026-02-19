"""
Configuration file for segmentation fine-tuning.
All hyperparameters and settings are centralized here.
"""

import os
import copy

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
batch_size = 12  # Batch size for training
lr_pretrain = 1e-3  # Learning rate for fine-tuning
latent_dim = 64  # Dimension of latent representation space
initial_epoch = 0  # Starting epoch (for resuming training)
num_epochs = 250  # Total number of training epochs

# Model architecture options
partial_decoder = 0  # Use partial decoder (0=full, 1=partial)
warm_start = 0  # Warm start from previous checkpoint

# =============================================================================
# TASK PARAMETERS
# =============================================================================
# Task type: 'volume' (3 classes) or 'cortex' (5 classes)
task = 'volume'  # 'volume' for whole kidney, 'cortex' for cortex+medulla

# Number of output classes (automatically set based on task)
num_classes = 3 if task == 'volume' else 5

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
# Get user's home directory for default paths
home_dir = "/add/path/to/your/project"  # todo: Update with your home directory

# Base directory containing resampled NIfTI files
data_dir = f'{home_dir}/Renal_fMRI/data/resampled'

# Cross-validation fold (1-5)
fold = 5

# Subject split files for current fold
val_sub = f'{home_dir}/Renal_fMRI/data/folds/test_f{fold}.npy'
train_sub = f'{home_dir}/Renal_fMRI/data/folds/train_f{fold}.npy'

# Base directory for saving results
base_save_dir = f'{home_dir}/Renal_fMRI/logs_paper/segmentation/folds/fold{fold}'

# Path to pre-trained weights for encoder initialization
checkpoint_path = f'{home_dir}/Renal_fMRI/checkpoints/pretrain.hdf5'

# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================
# MRI contrasts to use for fine-tuning
# Options: "ASL", "BOLD", "DIXON", "Diffusion", "T1_mapping_VIBE", 
#          "T1_mapping_fl2d", "T2_mapping_PREP"
contrast = ["ASL", "BOLD", "DIXON", "Diffusion", "T1_mapping_VIBE",
            "T1_mapping_fl2d", "T2_mapping_PREP"]

# Pre-training method identifier (for logging)
pretrain_method = 'BOLD'

# Target contrast for experiment naming
tgt_contrast = contrast[0] if len(contrast) == 1 else 'all'

# Experiment name
experiment_name = f'segmentation_{tgt_contrast}_kidney_volumes_nc_{num_classes}'

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
full_save_dir = os.path.join(base_save_dir, experiment_name)


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

    # Check task type
    assert task in ['volume', 'cortex'], f"Task must be 'volume' or 'cortex', got {task}"

    # Check number of classes
    expected_classes = 3 if task == 'volume' else 5
    assert num_classes == expected_classes, \
        f"For task '{task}', num_classes should be {expected_classes}, got {num_classes}"

    # Check fold number
    assert 1 <= fold <= 5, f"Fold must be between 1 and 5, got {fold}"

    # Check contrast list
    assert len(contrast) > 0, "At least one contrast must be specified"

    print("✓ Configuration validation passed")
    return True


# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================
def print_config_summary():
    """Print a formatted summary of the configuration."""

    print("\n" + "=" * 60)
    print("SEGMENTATION FINE-TUNING CONFIGURATION")
    print("=" * 60)

    print("\n DATA PARAMETERS:")
    print(f"  • Contrast(s): {contrast}")
    print(f"  • Image size: {img_size_x} x {img_size_y}")
    print(f"  • Dataset: {dataset}")
    print(f"  • Channels: {num_channels}")

    print("\n TRAINING PARAMETERS:")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Learning rate: {lr_pretrain}")
    print(f"  • Latent dimension: {latent_dim}")
    print(f"  • Epochs: {num_epochs}")

    print("\n TASK PARAMETERS:")
    print(f"  • Task: {task}")
    print(f"  • Number of classes: {num_classes}")
    print(f"  • Fold: {fold}")

    print("\n PATHS:")
    print(f"  • Data directory: {data_dir}")
    print(f"  • Save directory: {full_save_dir}")
    print(f"  • Training subjects: {train_sub}")
    print(f"  • Validation subjects: {val_sub}")
    print(f"  • Pre-trained checkpoint: {checkpoint_path}")

    print("\n EXPERIMENT:")
    print(f"  • Experiment name: {experiment_name}")
    print(f"  • Pre-training method: {pretrain_method}")

    print("\n" + "=" * 60 + "\n")


# =============================================================================
# CONVENIENCE FUNCTIONS FOR DIFFERENT TASKS
# =============================================================================

def get_config_for_task(task_name: str, fold_num: int = None, custom_params: dict = None):
    """
    Get configuration for a specific task with optional custom parameters.

    Args:
        task_name: Task name ('volume' or 'cortex')
        fold_num: Fold number (1-5), uses default if None
        custom_params: Dictionary of parameters to override

    Returns:
        Modified configuration dictionary
    """
    # Create a copy of the current configuration

    config = copy.deepcopy(globals())

    # Update task
    config['task'] = task_name
    config['num_classes'] = 3 if task_name == 'volume' else 5

    # Update fold if provided
    if fold_num is not None:
        config['fold'] = fold_num
        config['val_sub'] = f'{home_dir}/Renal_fMRI/data/folds/test_f{fold_num}.npy'
        config['train_sub'] = f'{home_dir}/Renal_fMRI/data/folds/train_f{fold_num}.npy'
        config['base_save_dir'] = f'{home_dir}/Renal_fMRI/logs_paper/segmentation/folds/fold{fold_num}'

    # Update experiment name
    config['tgt_contrast'] = config['contrast'][0] if len(config['contrast']) == 1 else 'all'
    config['experiment_name'] = f'seg_{config["tgt_contrast"]}_nc_{config["num_classes"]}_no_pretraining'
    config['full_save_dir'] = os.path.join(config['base_save_dir'], config['experiment_name'])

    # Apply custom parameters if provided
    if custom_params:
        for key, value in custom_params.items():
            if key in config:
                config[key] = value

    return config


def get_config_with_pretraining(pretrain_path: str, task_name: str = None, custom_params: dict = None):
    """
    Get configuration with a specific pre-trained weights path.

    Args:
        pretrain_path: Path to pre-trained weights
        task_name: Optional task override
        custom_params: Optional parameter overrides

    Returns:
        Modified configuration dictionary
    """
    config = copy.deepcopy(globals())
    config['checkpoint_path'] = pretrain_path

    if task_name is not None:
        config['task'] = task_name
        config['num_classes'] = 3 if task_name == 'volume' else 5

    # Update experiment name to indicate pretraining
    config['full_save_dir'] = os.path.join(config['base_save_dir'], config['experiment_name'])

    if custom_params:
        for key, value in custom_params.items():
            if key in config:
                config[key] = value

    return config


# =============================================================================
# PREDEFINED CONFIGURATIONS
# =============================================================================

# Configuration for whole kidney segmentation (3 classes)
VOLUME_CONFIG = {
    'task': 'volume',
    'num_classes': 3,
    'batch_size': 12,
    'num_epochs': 250,
    'experiment_name': 'seg_volume_nc_3_no_pretraining'
}

# Configuration for cortex/medulla segmentation (5 classes)
CORTEX_CONFIG = {
    'task': 'cortex',
    'num_classes': 5,
    'batch_size': 8,  # Smaller batch size for more classes
    'num_epochs': 300,  # More epochs for harder task
    'experiment_name': 'seg_cortex_nc_5_no_pretraining'
}

# Configuration for single contrast (BOLD)
BOLD_SEG_CONFIG = {
    'contrast': ['BOLD'],
    'batch_size': 12,
    'experiment_name': 'seg_BOLD_nc_3_no_pretraining'
}



# =============================================================================
# RUN VALIDATION ON IMPORT
# =============================================================================
# Validate configuration when this file is imported
if __name__ != "__main__":
    validate_config()
    print_config_summary()