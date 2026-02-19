"""
Constraint Map Generation for Constrained Contrastive Learning (CCL)

This script generates constraint maps from multi-parametric MRI data by:
1. Loading multi-contrast images and corresponding segmentation masks
2. Applying PCA for dimensionality reduction and denoising
3. Clustering the PCA components using K-means to create constraint maps
4. Saving the constraint maps for later use in contrastive learning pre-training

The constraint maps capture tissue-specific patterns and are used to guide
the contrastive learning process by enforcing that similar tissue regions
have similar feature representations.
"""

import argparse
import glob
import logging
import os
import pathlib
from typing import Dict, List, Optional, Tuple, Union

import natsort
import nibabel as nib
import numpy as np
import scipy.io as sio
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import utility functions (assuming these exist in utils.py)
from .data_utils import contrastStretch, myCrop3D, normalize, performDenoising


class ConstraintMapGenerator:
    """
    Generates constraint maps from multi-parametric MRI data for contrastive learning.

    Constraint maps are created by:
    1. Applying PCA to reduce dimensionality and denoise the input images
    2. Clustering the PCA components using K-means
    3. The resulting clusters represent distinct tissue patterns that guide
       contrastive representation learning

    Attributes:
        data_dir (str): Directory containing the resampled NIfTI files
        save_dir (str): Directory where constraint maps will be saved
        contrast (str): MRI contrast to process (e.g., 'BOLD', 'T2_mapping_PREP')
        num_clusters (int): Number of clusters for K-means
        output_shape (Tuple[int, int]): Target shape for output images (height, width)
        num_pc_components (int): Number of principal components to retain
        random_state (int): Random seed for reproducibility
    """

    def __init__(
            self,
            data_dir: str,
            save_dir: str,
            contrast: str,
            num_clusters: int = 20,
            output_shape: Tuple[int, int] = (256, 256),
            num_pc_components: int = 4,
            random_state: int = 42,
    ):
        """
        Initialize the ConstraintMapGenerator.

        Args:
            data_dir: Directory containing the resampled NIfTI files
            save_dir: Directory where constraint maps will be saved
            contrast: MRI contrast to process (e.g., 'BOLD', 'T2_mapping_PREP')
            num_clusters: Number of clusters for K-means (default: 20)
            output_shape: Target shape for output images (height, width) (default: (256, 256))
            num_pc_components: Number of principal components to retain (default: 4)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.contrast = contrast
        self.num_clusters = num_clusters
        self.output_shape = output_shape
        self.num_pc_components = num_pc_components
        self.random_state = random_state

        # Create save directory if it doesn't exist
        self.constraint_save_dir = os.path.join(save_dir, contrast)
        pathlib.Path(self.constraint_save_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized ConstraintMapGenerator for contrast: {contrast}")
        logger.info(f"  Data directory: {data_dir}")
        logger.info(f"  Save directory: {self.constraint_save_dir}")
        logger.info(f"  Number of clusters: {num_clusters}")
        logger.info(f"  Output shape: {output_shape}")

    def process_all_subjects(self) -> None:
        """
        Process all subjects in the data directory to generate constraint maps.

        For each subject, checks if constraint map already exists and if both
        images and labels are present before generating.
        """
        # Get sorted list of all subjects
        subject_list = natsort.natsorted(os.listdir(self.data_dir))
        logger.info(f"Found {len(subject_list)} subjects to process")

        for subject_name in tqdm(subject_list, desc="Processing subjects"):
            logger.debug(f"Processing subject: {subject_name}")
            self._process_single_subject(subject_name)

    def _process_single_subject(self, subject_name: str) -> None:
        """
        Generate constraint map for a single subject.

        Args:
            subject_name: Name of the subject directory
        """
        # Define output path for constraint map
        save_filename = f'Constraint_map_{subject_name}_{self.num_clusters}.mat'
        save_path = os.path.join(self.constraint_save_dir, save_filename)

        # Skip if constraint map already exists
        if os.path.exists(save_path):
            logger.debug(f"Constraint map already exists for {subject_name}, skipping")
            return

        # Check if images and labels exist
        if not self._has_images_and_labels(subject_name):
            logger.warning(f"Subject {subject_name} missing images or labels, skipping")
            return

        try:
            # Load image and mask
            img, mask = self._load_subject_data(subject_name)

            logger.info(f"Generating constraint map for {subject_name} with K={self.num_clusters}")

            # Generate parametric clusters
            constraint_map = self._generate_parametric_clusters(img, mask)

            # Normalize the constraint map
            constraint_map_normalized = np.stack(
                [normalize(constraint_map[..., i]) for i in range(constraint_map.shape[-1])],
                axis=-1
            )

            # Save results
            self._save_constraint_map(save_path, constraint_map_normalized, img, mask)
            logger.info(f"Successfully saved constraint map for {subject_name}")

        except Exception as e:
            logger.error(f"Failed to process subject {subject_name}: {str(e)}")
            raise

    def _has_images_and_labels(self, subject_name: str) -> bool:
        """
        Check if subject has both images and labels for the specified contrast.

        Args:
            subject_name: Name of the subject directory

        Returns:
            True if both images and volume labels exist, False otherwise
        """
        img_pattern = os.path.join(self.data_dir, subject_name, f"{self.contrast}/imagesTr/*")
        lbl_pattern = os.path.join(self.data_dir, subject_name, f"{self.contrast}/labelsTr/*volume*")

        has_images = len(glob.glob(img_pattern)) > 0
        has_labels = len(glob.glob(lbl_pattern)) > 0

        return has_images and has_labels

    def _load_subject_data(self, subject_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess image and mask data for a subject.

        The function:
        1. Loads all available image files for the contrast
        2. Loads left/right kidney and cortex segmentation masks
        3. Creates a combined 5-class mask (background, left cortex, right cortex, 
           left medulla, right medulla)
        4. Applies cropping and contrast stretching

        Args:
            subject_name: Name of the subject directory

        Returns:
            Tuple of (image_data, mask_data) as numpy arrays
        """
        logger.info(f"Loading {self.contrast} image for {subject_name}")

        # Load image data
        img_paths = glob.glob(os.path.join(self.data_dir, subject_name, f"{self.contrast}/imagesTr/*"))
        img_data = self._load_and_preprocess_images(img_paths)

        # Load mask data
        mask_data = self._load_and_combine_masks(subject_name)

        # Crop if necessary
        if img_data.shape[0] > self.output_shape[0] or img_data.shape[1] > self.output_shape[1]:
            img_data = myCrop3D(img_data, self.output_shape)
            mask_data = myCrop3D(mask_data, self.output_shape)

        # Apply contrast stretching
        img_data = contrastStretch(img_data, mask_data, 0.01, 99.9)

        logger.info(f"Loaded image shape: {img_data.shape}, mask shape: {mask_data.shape}")
        return img_data, mask_data

    def _load_and_preprocess_images(self, image_paths: List[str]) -> np.ndarray:
        """
        Load and preprocess multiple image files.

        Args:
            image_paths: List of paths to NIfTI image files

        Returns:
            Stacked and normalized image array
        """
        images = []
        for path in image_paths:
            # Load, flip, and rotate to standard orientation
            img = nib.load(path).get_fdata()
            img = np.flip(np.rot90(img, -1), 1)
            images.append(img)

        # Stack along last dimension (contrast/channel dimension)
        img_stack = np.stack(images, axis=-1)

        # Remove singleton dimension if present
        if img_stack.shape[-1] == 1:
            img_stack = np.squeeze(img_stack, axis=-1)

        # Normalize
        img_stack = normalize(img_stack)

        return img_stack

    def _load_and_combine_masks(self, subject_name: str) -> np.ndarray:
        """
        Load individual mask files and combine into a single multi-class mask.

        Creates a 5-class mask with the following classes:
        - Class 0: Background (later remapped to 5)
        - Class 1: Left cortex
        - Class 2: Right cortex
        - Class 3: Left medulla
        - Class 4: Right medulla

        Args:
            subject_name: Name of the subject directory

        Returns:
            Combined mask array with class labels
        """
        # Get paths to mask files
        mask_volume_paths = glob.glob(os.path.join(
            self.data_dir, subject_name, f"{self.contrast}/labelsTr/*volume*"
        ))
        mask_cortex_paths = glob.glob(os.path.join(
            self.data_dir, subject_name, f"{self.contrast}/labelsTr/*cortex*"
        ))

        # Find left and right masks
        path_vol_left = self._find_mask_by_side(mask_volume_paths, "left")
        path_vol_right = self._find_mask_by_side(mask_volume_paths, "right")
        path_cortex_left = self._find_mask_by_side(mask_cortex_paths, "left")
        path_cortex_right = self._find_mask_by_side(mask_cortex_paths, "right")

        # Load and preprocess each mask
        vol_left = self._load_mask(path_vol_left)
        vol_right = self._load_mask(path_vol_right)
        cortex_left = self._load_mask(path_cortex_left)
        cortex_right = self._load_mask(path_cortex_right)

        # Calculate medulla masks (kidney volume minus cortex)
        medulla_left = vol_left - cortex_left
        medulla_right = vol_right - cortex_right

        # Background mask (everything outside both kidneys)
        background = np.ones_like(vol_left) - vol_left - vol_right

        # Stack masks along last dimension and get class labels via argmax
        mask_stack = np.stack(
            (background, cortex_left, cortex_right, medulla_left, medulla_right),
            axis=-1
        )
        combined_mask = np.argmax(mask_stack, axis=-1)

        # Remap background from 0 to 5
        combined_mask[combined_mask == 0] = 5

        return combined_mask

    @staticmethod
    def _find_mask_by_side(mask_paths: List[str], side: str) -> str:
        """
        Find mask path containing the specified side (left/right).

        Args:
            mask_paths: List of mask file paths
            side: 'left' or 'right'

        Returns:
            Path to the matching mask

        Raises:
            ValueError: If no matching mask is found
        """
        matching = [s for s in mask_paths if side.lower() in s.lower()]
        if not matching:
            raise ValueError(f"No {side} mask found in {mask_paths}")
        return matching[0]

    @staticmethod
    def _load_mask(mask_path: str) -> np.ndarray:
        """
        Load and preprocess a single mask file.

        Args:
            mask_path: Path to NIfTI mask file

        Returns:
            Preprocessed mask array
        """
        mask = nib.load(mask_path).get_fdata()
        return np.flip(np.rot90(mask, -1), 1)

    def _generate_parametric_clusters(
            self,
            image_volume: np.ndarray,
            mask: np.ndarray,
    ) -> np.ndarray:
        """
        Generate parametric clusters using PCA and K-means.

        Steps:
        1. Apply PCA for dimensionality reduction and denoising
        2. Create a weight map focusing on kidney regions
        3. Apply total variation denoising to PCA components
        4. Cluster the denoised PCA components using K-means

        Args:
            image_volume: Input 4D image volume (H x W x D x C)
            mask: Segmentation mask (H x W x D)

        Returns:
            Parametric cluster map (H x W x D)

        Raises:
            AssertionError: If image_volume is not 4D
        """
        assert len(image_volume.shape) == 4, f"Expected 4D image, got shape {image_volume.shape}"

        x_dim, y_dim, z_dim, t_dim = image_volume.shape

        # Step 1: PCA for dimensionality reduction and denoising
        logger.debug(f"Applying PCA with {self.num_pc_components} components")
        flattened = np.reshape(image_volume, (-1, t_dim))
        pca = PCA(n_components=self.num_pc_components, random_state=self.random_state)
        pca_components = pca.fit_transform(flattened)
        pca_components = np.reshape(pca_components, (x_dim, y_dim, z_dim, self.num_pc_components))
        pca_components = normalize(pca_components)

        # Step 2: Create weight map focusing on kidney regions
        weight_map = self._create_weight_map(mask, margin=5)

        # Step 3: Apply total variation denoising to PCA components
        for idx in range(self.num_pc_components):
            denoised = performDenoising(pca_components[..., idx], wts=5)
            pca_components[..., idx] = weight_map * denoised

        # Step 4: K-means clustering
        logger.debug(f"Applying K-means clustering with {self.num_clusters} clusters")
        flattened_pca = np.reshape(pca_components, (-1, self.num_pc_components))
        kmeans = MiniBatchKMeans(
            n_clusters=self.num_clusters,
            random_state=self.random_state,
            batch_size=10000
        )
        cluster_labels = kmeans.fit_predict(flattened_pca)

        # Reshape back to original dimensions
        param_clusters = np.reshape(cluster_labels, (x_dim, y_dim, z_dim))

        return param_clusters

    def _create_weight_map(self, mask: np.ndarray, margin: int = 5) -> np.ndarray:
        """
        Create a weight map that focuses on kidney regions.

        The weight map is the average of bounding box masks for each slice,
        which gives higher weight to regions containing kidneys.

        Args:
            mask: Segmentation mask
            margin: Margin to add around bounding box

        Returns:
            Weight map array
        """
        # Create bounding box mask for each slice
        bbox_masks = []
        for slice_idx in range(mask.shape[-1]):
            bbox_mask = self._create_bounding_box_mask(mask[..., slice_idx], margin)
            bbox_masks.append(bbox_mask)

        # Stack and average
        bbox_stack = np.stack(bbox_masks, axis=-1)
        return bbox_stack

    @staticmethod
    def _create_bounding_box_mask(mask_slice: np.ndarray, margin: int = 5) -> np.ndarray:
        """
        Create a binary mask of the bounding box around a segmentation mask slice.

        Args:
            mask_slice: 2D binary segmentation mask (values: 0 or 1)
            margin: Margin to add around the bounding box

        Returns:
            Binary mask of the same shape as input with the bounding box region set to 1
        """
        # Find indices of non-zero elements
        rows, cols = np.nonzero(mask_slice)

        # If mask is empty, return zeros
        if len(rows) == 0 or len(cols) == 0:
            return np.zeros_like(mask_slice, dtype=np.uint8)

        # Calculate bounding box coordinates
        x_min, x_max = cols.min(), cols.max()
        y_min, y_max = rows.min(), rows.max()

        # Add margin (clipped to image boundaries)
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(mask_slice.shape[1] - 1, x_max + margin)
        y_max = min(mask_slice.shape[0] - 1, y_max + margin)

        # Create bounding box mask
        bbox_mask = np.zeros_like(mask_slice, dtype=np.uint8)
        bbox_mask[y_min:y_max + 1, x_min:x_max + 1] = 1

        return bbox_mask

    def _save_constraint_map(
            self,
            save_path: str,
            constraint_map: np.ndarray,
            original_image: np.ndarray,
            mask: np.ndarray
    ) -> None:
        """
        Save constraint map along with original image and mask to .mat file.

        Args:
            save_path: Path where to save the .mat file
            constraint_map: Generated constraint map
            original_image: Original image data
            mask: Segmentation mask
        """
        save_data = {
            'param': constraint_map,
            'img': original_image,
            'mask': mask
        }
        sio.savemat(save_path, save_data)
        logger.debug(f"Saved constraint map to {save_path}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate constraint maps for Constrained Contrastive Learning (CCL)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory where constraint maps will be saved"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the resampled NIfTI files"
    )

    parser.add_argument(
        "--contrast",
        type=str,
        required=True,
        choices=["BOLD", "T2_mapping_PREP", "T1_mapping_VIBE", "ASL", "Diffusion"],
        help="MRI contrast to process"
    )

    parser.add_argument(
        "--num_clusters",
        type=int,
        default=20,
        help="Number of clusters for K-means (default: 20)"
    )

    parser.add_argument(
        "--output_size",
        type=int,
        default=256,
        help="Output image size (height/width) (default: 256)"
    )

    parser.add_argument(
        "--num_pc_components",
        type=int,
        default=4,
        help="Number of principal components to retain (default: 4)"
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    # Validate arguments
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory does not exist: {args.data_dir}")

    # Create generator instance
    generator = ConstraintMapGenerator(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        contrast=args.contrast,
        num_clusters=args.num_clusters,
        output_shape=(args.output_size, args.output_size),
        num_pc_components=args.num_pc_components,
        random_state=args.random_seed
    )

    # Process all subjects
    generator.process_all_subjects()

    logger.info("Constraint map generation completed successfully")


if __name__ == "__main__":
    logger.info("Starting constraint map generation")
    main()