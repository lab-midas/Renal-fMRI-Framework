#!/usr/bin/env python
"""
Preprocess data for registration training.

This script converts the resampled NIfTI data structure into HDF5 files
that can be efficiently loaded by the registration data loader.

The HDF5 files contain:
    - 'contrasts/{contrast_name}/img': Image data for each contrast
    - 'contrasts/{contrast_name}/mask': Segmentation mask for each contrast
    - 'non_zero': List of slice indices that contain kidney tissue

Usage:
    python generate_reg_data.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR] [--debug]

Example:
    python generate_reg_data.py \
        --input_dir /path/to/input/data/ \
        --output_dir path/to/output/data/ \
        --debug
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import glob

import numpy as np
import nibabel as nib
import h5py
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from utils import setup_logging, get_logger

# Configure logging
logger = get_logger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# List of all possible contrasts
ALL_CONTRASTS = [
    "ASL",
    "BOLD",
    "DIXON",
    "Diffusion",
    "T1_mapping_VIBE",
    "T1_mapping_fl2d",
    "T2_mapping_PREP"
]

# Contrasts that typically have multiple channels (e.g., multi-echo)
MULTI_CHANNEL_CONTRASTS = ["BOLD", "Diffusion", "T1_mapping_VIBE"]


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def find_contrast_files(subject_dir: str, contrast: str) -> Tuple[List[str], List[str]]:
    """
    Find all image and mask files for a given contrast in a subject directory.

    Args:
        subject_dir: Path to subject directory
        contrast: Contrast name (e.g., 'BOLD', 'DIXON')

    Returns:
        Tuple of (image_paths, mask_paths)
    """
    contrast_dir = os.path.join(subject_dir, contrast)

    # Find image files
    img_pattern = os.path.join(contrast_dir, "imagesTr", "*.nii*")
    image_paths = sorted(glob.glob(img_pattern))

    # Find mask files
    mask_pattern = os.path.join(contrast_dir, "labelsTr", "*.nii*")
    all_masks = sorted(glob.glob(mask_pattern))

    # Separate left and right masks
    left_masks = [m for m in all_masks if "left" in m.lower()]
    right_masks = [m for m in all_masks if "right" in m.lower()]

    # For registration, we need both left and right masks
    # We'll combine them into a single mask with values: 0=background, 1=left, 2=right
    mask_paths = left_masks + right_masks

    return image_paths, mask_paths


def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """
    Load a NIfTI image and apply standard preprocessing.

    Args:
        image_path: Path to NIfTI file

    Returns:
        Preprocessed image array
    """
    # Load NIfTI
    img = nib.load(image_path)
    data = img.get_fdata().astype(np.float32)

    # Apply same transformations as in training
    # Flip and rotate to match training orientation
    data = np.flip(np.rot90(data, -1), 1)

    return data


def load_and_combine_masks(mask_paths: List[str], expected_shape: Tuple) -> np.ndarray:
    """
    Load left and right masks and combine into a single multi-class mask.

    Args:
        mask_paths: List of mask file paths (should contain left and right masks)
        expected_shape: Expected shape of the mask

    Returns:
        Combined mask with values: 0=background, 1=left kidney, 2=right kidney
    """
    # Initialize combined mask
    combined_mask = np.zeros(expected_shape, dtype=np.float32)

    for mask_path in mask_paths:
        # Load mask
        mask = nib.load(mask_path)
        mask_data = mask.get_fdata().astype(np.float32)

        # Apply same transformations
        mask_data = np.flip(np.rot90(mask_data, -1), 1)

        # Assign class based on side
        if "left" in mask_path.lower():
            combined_mask[mask_data > 0] = 1  # Left kidney = 1
        elif "right" in mask_path.lower():
            combined_mask[mask_data > 0] = 2  # Right kidney = 2

    return combined_mask


def find_non_zero_slices(mask: np.ndarray) -> List[int]:
    """
    Find slice indices that contain kidney tissue.

    Args:
        mask: 3D mask array (D, H, W) with values 0, 1, 2

    Returns:
        List of slice indices with non-zero values
    """
    non_zero_slices = []
    for slice_idx in range(mask.shape[0]):
        if np.any(mask[slice_idx] > 0):
            non_zero_slices.append(slice_idx)
    return non_zero_slices


def process_subject(
        subject_dir: str,
        subject_id: str,
        output_dir: str,
        contrasts: List[str]
) -> bool:
    """
    Process a single subject and save as HDF5 file.

    Args:
        subject_dir: Path to subject directory
        subject_id: Subject ID (e.g., 'P_01_A')
        output_dir: Output directory for HDF5 files
        contrasts: List of contrasts to process

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Processing subject: {subject_id}")

    # Initialize dictionary to store data for this subject
    subject_data = {}
    all_non_zero = []

    # Process each contrast
    for contrast in contrasts:
        logger.debug(f"  Processing contrast: {contrast}")

        # Find files for this contrast
        img_paths, mask_paths = find_contrast_files(subject_dir, contrast)

        if not img_paths:
            logger.warning(f"    No images found for {contrast}, skipping")
            continue

        # Load all images for this contrast
        images = []
        for img_path in img_paths:
            img_data = load_and_preprocess_image(img_path)
            images.append(img_data)

        # Stack images along channel dimension
        if len(images) == 1:
            # Single channel
            img_stack = images[0]
            if img_stack.ndim == 3:
                # Add channel dimension: (H, W, D) -> (D, H, W, 1)
                img_stack = np.transpose(img_stack, (2, 0, 1))[..., np.newaxis]
            else:
                logger.error(f"Unexpected image dimensions: {img_stack.shape}")
                return False
        else:
            # Multiple channels (e.g., multi-echo BOLD)
            # Each image should be (H, W, D)
            # Stack along new axis: (H, W, D, C)
            img_stack = np.stack(images, axis=-1)
            # Transpose to (D, H, W, C)
            img_stack = np.transpose(img_stack, (2, 0, 1, 3))

        logger.debug(f"    Image shape after stacking: {img_stack.shape}")

        # Load and combine masks
        if mask_paths:
            # Get expected shape from image (spatial dimensions only)
            expected_mask_shape = img_stack.shape[1:3] + (img_stack.shape[0],)
            mask_combined = load_and_combine_masks(mask_paths, expected_mask_shape)

            # Ensure mask has same orientation as image (D, H, W)
            if mask_combined.ndim == 3:
                # Already (D, H, W) from load_and_combine_masks
                pass
            else:
                logger.error(f"Unexpected mask dimensions: {mask_combined.shape}")
                return False

            logger.debug(f"    Mask shape: {mask_combined.shape}")

            # Find non-zero slices from mask
            non_zero = find_non_zero_slices(mask_combined)
            all_non_zero.extend(non_zero)
        else:
            logger.warning(f"    No masks found for {contrast}")
            mask_combined = np.zeros(img_stack.shape[:3], dtype=np.float32)
            non_zero = []

        # Store in subject data dictionary
        subject_data[contrast] = {
            'img': img_stack,
            'mask': mask_combined
        }

    # If no data was processed, return False
    if not subject_data:
        logger.warning(f"  No valid contrasts found for {subject_id}")
        return False

    # Get unique non-zero slices (union across all contrasts)
    unique_non_zero = sorted(set(all_non_zero))

    # Save to HDF5
    output_file = os.path.join(output_dir, f"{subject_id}.h5")

    try:
        with h5py.File(output_file, 'w') as f:
            # Create contrasts group
            contrasts_group = f.create_group('contrasts')

            # Save data for each contrast
            for contrast_name, contrast_data in subject_data.items():
                contrast_group = contrasts_group.create_group(contrast_name)

                # Save image data
                contrast_group.create_dataset(
                    'img',
                    data=contrast_data['img'],
                    compression='gzip',
                    compression_opts=4
                )

                # Save mask data
                contrast_group.create_dataset(
                    'mask',
                    data=contrast_data['mask'],
                    compression='gzip',
                    compression_opts=4
                )

            # Save non-zero slice indices
            f.create_dataset('non_zero', data=unique_non_zero, dtype='int32')

        logger.info(f"  ✓ Saved to {output_file}")
        logger.info(f"    Image shapes: { {k: v['img'].shape for k, v in subject_data.items()} }")
        logger.info(f"    Non-zero slices: {len(unique_non_zero)}")

        return True

    except Exception as e:
        logger.error(f"  ✗ Error saving to HDF5: {e}")
        return False


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess data for registration')

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing resampled data')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for HDF5 files')

    parser.add_argument('--contrasts', type=str, nargs='+',
                        default=ALL_CONTRASTS,
                        help=f'Contrasts to process (default: all)')

    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode (process only first subject)')

    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing HDF5 files')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("REGISTRATION DATA PREPROCESSING")
    logger.info("=" * 60)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Contrasts to process: {args.contrasts}")
    logger.info(f"Debug mode: {args.debug}")
    logger.info(f"Overwrite: {args.overwrite}")

    # Find all subjects
    subject_dirs = sorted(glob.glob(os.path.join(args.input_dir, "*")))
    subject_ids = [os.path.basename(d) for d in subject_dirs if os.path.isdir(d)]

    logger.info(f"Found {len(subject_ids)} subjects")

    if args.debug:
        subject_ids = subject_ids[:3]
        logger.info(f"Debug mode: processing first {len(subject_ids)} subjects")

    # Statistics
    successful = 0
    failed = 0
    skipped = 0

    # Process each subject
    for subject_id in tqdm(subject_ids, desc="Processing subjects"):
        subject_dir = os.path.join(args.input_dir, subject_id)
        output_file = os.path.join(args.output_dir, f"{subject_id}.h5")

        # Check if output already exists
        if os.path.exists(output_file) and not args.overwrite:
            logger.debug(f"Skipping {subject_id} (output file exists)")
            skipped += 1
            continue

        # Process subject
        success = process_subject(
            subject_dir=subject_dir,
            subject_id=subject_id,
            output_dir=args.output_dir,
            contrasts=args.contrasts
        )

        if success:
            successful += 1
        else:
            failed += 1

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total subjects: {len(subject_ids)}")
    logger.info(f"Successfully processed: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped (already exist): {skipped}")
    logger.info(f"Output directory: {args.output_dir}")

    # Verify output files
    logger.info("\nVerifying output files...")
    output_files = sorted(glob.glob(os.path.join(args.output_dir, "*.h5")))
    logger.info(f"Found {len(output_files)} HDF5 files")

    # Check first file as example
    if output_files:
        example_file = output_files[0]
        logger.info(f"\nExample file: {os.path.basename(example_file)}")

        try:
            with h5py.File(example_file, 'r') as f:
                logger.info("  Contents:")
                logger.info(f"    non_zero: {f['non_zero'][:10]}... (shape: {f['non_zero'].shape})")

                contrasts_group = f['contrasts']
                for contrast_name in contrasts_group.keys():
                    contrast_group = contrasts_group[contrast_name]
                    img_shape = contrast_group['img'].shape
                    mask_shape = contrast_group['mask'].shape
                    logger.info(f"    {contrast_name}: img{img_shape}, mask{mask_shape}")
        except Exception as e:
            logger.error(f"  Error reading example file: {e}")


# =============================================================================
# UTILITY FUNCTIONS FOR VERIFICATION
# =============================================================================

def verify_h5_file(file_path: str) -> Dict:
    """
    Verify the contents of an HDF5 file.

    Args:
        file_path: Path to HDF5 file

    Returns:
        Dictionary with file information
    """
    info = {
        'file': os.path.basename(file_path),
        'contrasts': [],
        'non_zero_count': 0,
        'shapes': {}
    }

    try:
        with h5py.File(file_path, 'r') as f:
            # Get non-zero slices
            info['non_zero_count'] = len(f['non_zero'][:])

            # Get contrasts
            contrasts_group = f['contrasts']
            for contrast_name in contrasts_group.keys():
                info['contrasts'].append(contrast_name)

                img = contrasts_group[contrast_name]['img']
                mask = contrasts_group[contrast_name]['mask']

                info['shapes'][contrast_name] = {
                    'img': img.shape,
                    'mask': mask.shape
                }
    except Exception as e:
        info['error'] = str(e)

    return info


def verify_all_files(output_dir: str):
    """
    Verify all HDF5 files in a directory.

    Args:
        output_dir: Directory containing HDF5 files
    """
    h5_files = sorted(glob.glob(os.path.join(output_dir, "*.h5")))

    print(f"\nVerifying {len(h5_files)} HDF5 files...")

    for h5_file in h5_files:
        info = verify_h5_file(h5_file)

        if 'error' in info:
            print(f"✗ {info['file']}: ERROR - {info['error']}")
        else:
            print(f"✓ {info['file']}: {info['non_zero_count']} non-zero slices, "
                  f"contrasts: {info['contrasts']}")


if __name__ == "__main__":
    main()

    # Uncomment to verify files after processing
    # verify_all_files('/path/to/data')