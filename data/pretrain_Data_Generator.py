"""
Data loader for Constrained Contrastive Learning (CCL) pre-training.

This module provides a custom Keras Sequence data generator that loads constraint maps
and prepares batches for contrastive learning. It handles:
- Loading constraint maps from .mat files
- Generating patches for contrastive learning
- Creating constraint masks for loss calculation
- Shuffling and batching data efficiently
"""

import itertools
import os
import threading
from random import shuffle
from typing import List, Tuple, Optional, Any

import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from skimage.util import view_as_blocks


class DataLoaderObj(Sequence):
    """
    Custom data generator for Constrained Contrastive Learning (CCL).

    This class loads constraint maps and prepares batches of image patches
    with corresponding cluster labels and masks for contrastive learning.

    Attributes:
        patch_size (int): Size of patches for patch-wise contrastive learning
        batch_size (int): Number of samples per batch
        num_channels (int): Number of image channels
        cfg: Configuration object containing all parameters
        ft_param (bool): Whether using fine-tuning parameters
        contrast (list): List of MRI contrasts to load
        num_constraints (int): Number of constraint maps (2 if using mask sampling)
        num_clusters (int): Number of clusters for sampling
        train_flag (bool): Whether this is training or validation loader
        input_img (list): List of subject IDs
        data_dir (str): Directory containing constraint maps
        img (list): Loaded image data
        param_cluster (list): Loaded cluster maps
        ft (list): Optional fine-tuning parameters
        img_size_x, img_size_y (int): Image dimensions
        arr_indexes (list): List of (image_idx, slice_idx, channel_idx) combinations
    """

    def __init__(self, cfg: Any, debug: bool, train_flag: bool = True):
        """
        Initialize the data loader.

        Args:
            cfg: Configuration object containing all parameters
            debug: If True, load only first 3 subjects for debugging
            train_flag: If True, load training subjects; else load validation subjects
        """
        # Store configuration parameters
        self.patch_size = cfg.patch_size
        self.batch_size = cfg.batch_size
        self.num_channels = cfg.num_channels
        self.cfg = cfg
        self.ft_param = cfg.ft_param
        self.contrast = cfg.contrast
        # Number of constraints: mask + cluster map if using mask sampling, else just cluster map
        self.num_constraints = 2 if cfg.use_mask_sampling else 1
        self.num_clusters = cfg.num_samples_loss_eval
        self.train_flag = train_flag

        # Load subject list based on train/validation flag
        if train_flag:
            print('Initializing training dataloader')
            self.input_img = np.load(cfg.train_sub).tolist()
        else:
            print('Initializing validation dataloader')
            self.input_img = np.load(cfg.val_sub).tolist()

        # Debug mode: use only first 3 subjects
        if debug:
            print(f"Debug mode: loading only first 3 of {len(self.input_img)} subjects")
            self.input_img = self.input_img[:3]

        # Data directories and storage
        self.data_dir = cfg.data_dir
        self.img = []  # Store image data
        self.param_cluster = []  # Store cluster maps
        if self.ft_param:
            self.ft = []  # Store fine-tuning parameters if needed

        # Load all data
        print(f'### Number of subjects: {len(self.input_img)}')
        print('### Started loading images...')

        if self.ft_param:
            self.load_ft()  # Load with fine-tuning parameters
        else:
            self.load_img_cluster()  # Load only images and clusters

        print('### Parameters and images loaded successfully')

        # Calculate total number of samples (slices × channels)
        self.len_of_data = sum([img.shape[0] * img.shape[-1] for img in self.img])
        self.num_samples = self.len_of_data // 1  # Use it to control size of samples per epoch
        print(f'### Total samples available: {self.num_samples}')

        # Image dimensions
        self.img_size_x = cfg.img_size_x
        self.img_size_y = cfg.img_size_y

        # Generate all possible (image, slice, channel) combinations
        self.arr_indexes = self.get_indexes()
        print(f'### Generated {len(self.arr_indexes)} index combinations')

    def load_ft(self) -> None:
        """
        Load data with fine-tuning parameters.

        This method loads images, cluster maps, and fine-tuning parameters
        from .mat files. It handles both 3D and 4D arrays and applies
        necessary transpositions to standardize orientation.

        The .mat file contains:
            - 'img': Original image data
            - 'param': Constraint map (clustered PCA components)
            - 'mask': Segmentation mask
        """
        for sub in self.input_img:
            for parameter in self.contrast:
                # Construct path to constraint map file
                path = os.path.join(self.data_dir, parameter, f'Constraint_map_{sub}_20.mat')

                if os.path.exists(path):
                    # Load .mat file
                    f = sio.loadmat(path)

                    # Extract and convert data to float32
                    img = f['img'].astype('float32')  # Original image
                    ft = f['param'].astype('float32')  # Constraint map (fine-tuning)
                    mask = f['mask'].astype('float32')  # Segmentation mask

                    # Remap background class from 0 to 5 (to avoid conflict with other classes)
                    mask[mask == 0] = 5

                    # Transpose arrays to have slices as first dimension: (depth, height, width, channels)
                    # This makes it easier to iterate over slices
                    if img.ndim == 4:
                        img = np.transpose(img, (2, 0, 1, 3))  # (depth, H, W, channels)
                    else:
                        img = np.transpose(img, (2, 0, 1))[..., None]  # Add channel dimension

                    # Transpose fine-tuning parameters: (depth, H, W, channels)
                    ft = np.transpose(ft, (2, 0, 1, 3))

                    # Transpose mask and tile to match image channels
                    # mask shape becomes: (depth, H, W, channels)
                    mask = np.transpose(mask, (2, 0, 1))[..., None]
                    mask = np.tile(mask, (1, 1, 1, img.shape[-1]))

                    # Store in class lists
                    self.img.append(img)
                    self.param_cluster.append(mask)
                    self.ft.append(ft)
                else:
                    print(f"Warning: File not found - {path}")

    def load_img_cluster(self) -> None:
        """
        Load only images and cluster maps (no fine-tuning parameters).

        This is the standard loading method for pre-training.
        The .mat file contains:
            - 'img': Original image data
            - 'param': Constraint map (clustered PCA components)
        """
        for sub in self.input_img:
            for parameter in self.contrast:
                # Construct path to constraint map file
                path = os.path.join(self.data_dir, parameter, f'Constraint_map_{sub}_20.mat')

                if os.path.exists(path):
                    # Load .mat file
                    f = sio.loadmat(path)

                    # Extract and convert data to float32
                    img = f['img'].astype('float32')  # Original image
                    cluster_map = f['param'].astype('float32')  # Constraint map

                    # Transpose arrays to have slices as first dimension: (depth, height, width, channels)
                    if img.ndim == 4:
                        img = np.transpose(img, (2, 0, 1, 3))  # (depth, H, W, channels)
                    else:
                        img = np.transpose(img, (2, 0, 1))[..., None]  # Add channel dimension

                    # Transpose cluster map: (depth, H, W)
                    cluster_map = np.transpose(cluster_map, (2, 0, 1))

                    # Store in class lists
                    self.img.append(img)
                    self.param_cluster.append(cluster_map)
                else:
                    print(f"Warning: File not found - {path}")

    def get_indexes(self) -> List[Tuple[int, int, int]]:
        """
        Generate all possible (image_idx, slice_idx, channel_idx) combinations.

        Returns:
            List of tuples, each containing (image_index, slice_index, channel_index)
            that will be used to sample patches during training.
        """
        # Create list of (image_index, num_slices, num_channels) for each loaded image
        shapes = [(i, img.shape[0], img.shape[-1]) for i, img in enumerate(self.img)]

        # Generate all combinations of image, slice, and channel
        combinations = []
        for img_idx, num_slices, num_channels in shapes:
            for slice_idx, channel_idx in itertools.product(range(num_slices), range(num_channels)):
                combinations.append((img_idx, slice_idx, channel_idx))

        # Shuffle for random sampling
        shuffle(combinations)

        return combinations

    def __len__(self) -> int:
        """
        Return the number of batches per epoch.

        Returns:
            Number of batches (floor division of total indices by batch_size)
        """
        return len(self.arr_indexes) // self.batch_size

    def get_len(self) -> int:
        """
        Get total number of samples.

        Returns:
            Total number of index combinations
        """
        return len(self.arr_indexes)

    def on_epoch_end(self) -> None:
        """
        Called at the end of each epoch.
        Shuffles the indices to ensure different sample ordering each epoch.
        """
        shuffle(self.arr_indexes)

    def __shape__(self) -> Tuple:
        """
        Get the shape of a sample batch.

        Returns:
            Shape tuple of the first batch
        """
        data = self.__getitem__(0)
        return data.shape

    def __getitem__(self, idx: int) -> Tuple[tf.Tensor, Any]:
        """
        Get a batch of data for training.

        Args:
            idx: Batch index

        Returns:
            Tuple of (input_images, target_labels) where:
                - input_images: Tensor of shape (batch_size, H, W, channels)
                - target_labels: Either cluster maps or tuple of (clusters, ft_params)
        """
        with threading.Lock():  # Thread-safe data loading
            # Get indices for this batch
            batch_indices = self.arr_indexes[idx * self.batch_size: (idx + 1) * self.batch_size]

            # Generate input images
            x_train = self.generate_X(batch_indices)
            x_train = tf.identity(x_train)  # Convert to Tensor

            # Generate target labels based on whether fine-tuning is enabled
            if self.ft_param:
                # With fine-tuning: return both cluster maps and fine-tuning parameters
                y_param_cluster = self.generate_clusters(batch_indices, self.param_cluster)
                y_ft = self.generate_clusters(batch_indices, self.ft)
                y_train = (tf.identity(y_param_cluster), tf.identity(y_ft))
            else:
                # Standard pre-training: return only cluster maps
                y_train = self.generate_clusters(batch_indices, self.param_cluster)
                y_train = tf.identity(y_train)

            return x_train, y_train

    def generate_X(self, list_idx: List[Tuple[int, int, int]]) -> np.ndarray:
        """
        Generate input images for a batch.

        Args:
            list_idx: List of (image_idx, slice_idx, channel_idx) tuples

        Returns:
            Batch of images with shape (batch_size, H, W, channels)
        """
        X = np.zeros(
            (self.batch_size, self.img_size_x, self.img_size_y, self.num_channels),
            dtype="float64"
        )

        for jj in range(self.batch_size):
            # Unpack indices: (image_index, slice_index, channel_index)
            img_idx, slice_idx, channel_idx = list_idx[jj]

            # Extract the specific slice and channel, add channel dimension
            X[jj] = self.img[img_idx][slice_idx, ..., channel_idx][..., None]

        return X

    def generate_mask(self, X: np.ndarray) -> np.ndarray:
        """
        Generate a binary mask from foreground regions.

        The mask is created by thresholding: any pixel > minimum value is considered foreground.

        Args:
            X: Input image batch

        Returns:
            Binary mask with same shape as input
        """
        mask = np.zeros(X.shape)
        mask[X > X.min()] = 1  # Foreground = pixels above minimum intensity
        return mask

    def generate_clusters(
            self,
            list_idx: List[Tuple[int, int, int]],
            param: List[np.ndarray]
    ) -> np.ndarray:
        """
        Generate cluster labels for contrastive learning.

        This is the core function for CCL. It:
        1. Extracts cluster maps for each sample
        2. Creates patches and finds majority class in each patch
        3. Generates sampling masks for loss calculation

        Args:
            list_idx: List of (image_idx, slice_idx, channel_idx) tuples
            param: List of parameter maps (either clusters or fine-tuning params)

        Returns:
            Tensor of cluster labels with shape (batch_size, H/patch_size, W/patch_size, num_constraints)
        """
        patch_size = self.patch_size
        batch_size = self.batch_size
        num_constraints = self.num_constraints

        # Initialize arrays
        X = np.zeros((batch_size, self.img_size_x, self.img_size_y, self.num_channels), dtype="float64")

        # Y stores cluster labels and masks
        # Shape: (H, W, num_constraints, batch_size)
        # Note: batch dimension is transposed for view_as_blocks function
        Y = np.zeros((self.img_size_x, self.img_size_y, num_constraints, batch_size), dtype="int64")

        # Mask for foreground regions
        mask = np.zeros((self.img_size_x, self.img_size_y, batch_size), dtype="int64")

        # Fill arrays for each sample in batch
        for jj in range(batch_size):
            # Unpack indices
            img_idx, slice_idx, channel_idx = list_idx[jj]

            # Extract image slice
            temp = self.img[img_idx][slice_idx, ..., channel_idx][..., None]
            X[jj] = temp

            # Generate foreground mask
            mask[..., jj] = self.generate_mask(temp[..., 0])

            # Extract parameter map (clusters or fine-tuning)
            if param[img_idx].ndim == 3:
                # 3D case: (depth, H, W) - no channel dimension
                param_in = param[img_idx][slice_idx]
            else:
                # 4D case: (depth, H, W, channels) - select specific channel
                param_in = param[img_idx][slice_idx, ..., channel_idx]

            # Store cluster map (first constraint)
            Y[..., 0, jj] = np.squeeze(param_in)

            # Store mask (second constraint, if used)
            Y[..., -1, jj] = mask[..., jj]

        # Patch-based processing: identify majority class in each non-overlapping patch
        if patch_size > 1:
            # View Y as blocks of size patch_size x patch_size
            # Result shape: (H/patch_size, W/patch_size, patch_size, patch_size, num_constraints, batch_size)
            y_train_blk = view_as_blocks(Y, (patch_size, patch_size, num_constraints, batch_size))
            y_train_blk = y_train_blk.squeeze()  # Remove singleton dimensions

            # Get dimensions of patch grid
            xDim, yDim = y_train_blk.shape[0], y_train_blk.shape[1]

            # Reshape to combine patch pixels: (grid_x, grid_y, patch_size*patch_size, num_constraints, batch_size)
            y_train_blk = np.reshape(
                y_train_blk,
                (xDim, yDim, patch_size * patch_size, num_constraints, batch_size)
            )

            # Find majority class in each patch (along the patch pixels dimension)
            y_train = np.apply_along_axis(self.get_freq_labels, -3, y_train_blk)
        else:
            # No patching needed
            y_train = Y

        # Generate sampling indices from masks for loss calculation
        for jj in range(batch_size):
            # Extract mask for this batch item
            temp_mask = y_train[..., -1, jj].squeeze()

            # Generate random indices within mask for sampling
            y_train[..., -1, jj] = self.generate_indices_from_mask(temp_mask)

        # Transpose back to (batch_size, H/patch_size, W/patch_size, num_constraints)
        Y_final = np.transpose(y_train, (3, 0, 1, 2))

        return Y_final

    @staticmethod
    def get_freq_labels(arr: np.ndarray) -> int:
        """
        Find the most frequent label in an array.

        Used with apply_along_axis to find majority class in each patch.

        Args:
            arr: 1D array of labels

        Returns:
            Most frequent label
        """
        return np.bincount(arr).argmax()

    def generate_indices_from_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Generate a mask with random indices for patch-wise CCL loss calculation.

        This creates a binary mask where a subset of foreground pixels are
        selected as sampling points for the contrastive loss.

        Args:
            mask: Binary mask of foreground regions

        Returns:
            Binary mask of same shape with random sampling points set to 1
        """
        xDim, yDim = mask.shape

        # Flatten mask for easier indexing
        mask_flat = np.ndarray.flatten(mask.copy())

        # Find all foreground indices
        all_foreground_indices = np.where(mask_flat == 1)[0]

        # Randomly shuffle and select subset
        np.random.shuffle(all_foreground_indices)
        selected_indices = all_foreground_indices[:self.num_clusters]

        # Create new mask with only selected indices
        sampling_mask_flat = np.zeros(mask_flat.shape, dtype='int64')
        sampling_mask_flat[selected_indices] = 1

        # Reshape back to original dimensions
        sampling_mask = np.reshape(sampling_mask_flat, (xDim, yDim))

        return sampling_mask