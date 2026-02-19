"""
Data loader for affine registration training.

This module provides a custom Keras Sequence data generator that loads
multi-contrast images and corresponding masks from HDF5 files for
training the affine registration network.
"""

import os
import threading
from itertools import product
from random import shuffle, sample
from typing import List, Tuple, Dict, Any, Optional

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from scipy.ndimage import distance_transform_edt as edt
from tqdm import tqdm

from .data_utils import normalize


class DataLoaderRegistration(Sequence):
    """
    Custom data generator for affine registration training.

    This class loads multi-contrast images and masks from pre-processed HDF5 files
    and generates batches for training the registration network. It supports:
    - Multiple moving contrasts registered to a template contrast
    - Weighted sampling based on mask regions
    - Distance transform-based weighting maps

    Attributes:
        batch_size (int): Number of samples per batch
        debug (bool): Debug mode flag
        num_channels (int): Number of image channels
        cfg: Configuration object
        type_mask (str): Mask type ('bbox' or 'distance')
        out_features (bool): Whether to output features
        affine (bool): Whether using affine registration
        weighted (bool): Whether to use mask weighting
        num_classes (int): Number of classes in masks
        pca_template (bool): Whether to use PCA on template
        train_flag (bool): Training or validation mode
        size (tuple): Image dimensions (H, W)
        input_img (list): List of subject IDs
        data_dir (str): Directory containing HDF5 files
        out_labels (bool): Whether to output labels
        contrast_template (list): Template contrast(s)
        contrast_moving (list): Moving contrast(s)
        pairs (list): Loaded data pairs
        indexes (list): List of (pair_idx, slice_idx, combo_idx, template_idx) tuples
        img_size_x, img_size_y (int): Image dimensions
    """

    def __init__(self, cfg, debug: bool, train_flag: bool = True):
        """
        Initialize the data loader.

        Args:
            cfg: Configuration object
            debug: Debug mode flag
            train_flag: If True, load training data; else load validation data
        """
        self.batch_size = cfg.batch_size
        self.debug = debug
        self.num_channels = cfg.num_channels
        self.cfg = cfg
        self.type_mask = cfg.type_mask
        self.out_features = cfg.out_features
        self.affine = cfg.affine
        self.weighted = cfg.weighted
        self.num_classes = cfg.num_classes
        self.pca_template = cfg.pca_template
        self.train_flag = train_flag
        self.size = (cfg.img_size_x, cfg.img_size_y)

        # Load subject list based on train/validation flag
        if train_flag:
            print('Initializing training dataloader')
            subject_file = cfg.train_sub
            self.input_img = np.load(subject_file).tolist()
        else:
            print('Initializing validation dataloader')
            subject_file = cfg.val_sub
            self.input_img = np.load(subject_file).tolist()

        # Debug mode: use only first 3 subjects
        if self.debug:
            print(f"Debug mode: loading only first 3 of {len(self.input_img)} subjects")
            self.input_img = self.input_img[:3]

        # Data directories and storage
        self.data_dir = cfg.data_dir
        self.out_labels = cfg.out_labels
        self.contrast_template = cfg.contrast_template
        self.contrast_moving = cfg.contrast_moving

        # Storage for loaded data
        self.pairs = []  # List of dictionaries containing image/mask data
        self.indexes = []  # List of (pair_idx, slice_idx, combo_idx, template_idx)

        # Load all data
        print('### Loading data from HDF5 files...')
        self.load_imgs()

        # Shuffle indexes
        shuffle(self.indexes)
        shuffle(self.indexes)

        # Calculate dataset size
        self.len_of_data = len(self.indexes)
        self.num_samples = self.len_of_data
        print(f'### Total samples: {self.num_samples}')

        # Image dimensions
        self.img_size_x = cfg.img_size_x
        self.img_size_y = cfg.img_size_x

    def load_dict_from_h5py(self, filename: str) -> Tuple[Dict, List]:
        """
        Load data from HDF5 file.

        The HDF5 file structure should be:
        - 'non_zero': List of slices with signal
        - 'contrasts/{contrast_name}/img': Image data
        - 'contrasts/{contrast_name}/mask': Mask data

        Args:
            filename: Path to HDF5 file

        Returns:
            Tuple of (data_dict, non_zero_list) where:
                data_dict: Dictionary with contrast names as keys, each containing
                          'img' and 'mask' arrays
                non_zero_list: List of slice indices with signal
        """
        data_dict = {}

        with h5py.File(filename, 'r') as h5_file:
            # Load non-zero slice indices
            non_zero = h5_file['non_zero'][:].tolist()

            # Load data for each contrast
            nested_dicts_group = h5_file['contrasts']

            # Load moving contrasts
            for dict_name in self.contrast_moving:
                if dict_name in nested_dicts_group:
                    sub_dict = {}
                    for key in nested_dicts_group[dict_name].keys():
                        sub_dict[key] = nested_dicts_group[dict_name][key][()].astype(np.float32)
                    data_dict[dict_name] = sub_dict

            # Load template contrasts
            for dict_name in self.contrast_template:
                if dict_name in nested_dicts_group:
                    sub_dict = {}
                    for key in nested_dicts_group[dict_name].keys():
                        sub_dict[key] = nested_dicts_group[dict_name][key][()].astype(np.float32)
                    data_dict[dict_name] = sub_dict

        return data_dict, non_zero

    def load_imgs(self) -> None:
        """
        Load all images from HDF5 files and generate index combinations.

        For each subject, this method:
        1. Loads the HDF5 file
        2. Extracts image and mask data for all contrasts
        3. Generates all combinations of slices and channels for training
        4. Filters out slices without signal
        """
        pair_idx = 0

        for sub in tqdm(self.input_img, desc="Loading subjects"):
            filename = os.path.join(self.data_dir, f'{sub}.h5')

            if os.path.exists(filename):
                # Load data from HDF5
                dict_sub, non_zero_no_signal = self.load_dict_from_h5py(filename)
                self.pairs.append(dict_sub)

                # Get dimensions for each moving contrast
                dims = [dict_sub[c]['img'].shape[-1] for c in self.contrast_moving]

                # Number of slices (assume all contrasts have same number of slices)
                slices = dict_sub[self.contrast_template[0]]['img'].shape[0]

                # Generate channel combinations for multi-channel contrasts
                # Sample up to 4 channels per contrast if more are available
                ranges = []
                for n in dims:
                    if n < 5:
                        ranges.append(list(range(n)))
                    else:
                        ranges.append(sample(range(n), 4))  # Sample 4 channels

                all_combinations = list(product(*ranges))

                # Get template slices dimension
                template_slices = dict_sub[self.contrast_template[0]]['img'].shape[-1]

                # Find slices that have masks in all moving contrasts
                non_zero_list = [
                    np.any(dict_sub[c]['mask'] != 0, axis=(1, 2))
                    for c in self.contrast_moving
                ]
                non_zero_no_mask = np.all(non_zero_list, axis=0).tolist()

                # Combine with signal mask
                non_zero = [x and y for (x, y) in zip(non_zero_no_mask, non_zero_no_signal)]
                slices_range = np.arange(slices)
                slices_range = slices_range[non_zero]

                # Generate all combinations of (subject_idx, slice_idx, channel_combo, template_idx)
                for combo in all_combinations:
                    for s in slices_range:
                        for d in range(template_slices):
                            self.indexes.append((pair_idx, s, combo, d))

                pair_idx += 1
                print(f'  Generated {len(self.indexes)} samples so far')
            else:
                print(f'\n⚠ File not found: {filename}')

        print(f'\n### Successfully loaded {pair_idx} files')
        print(f'### Total samples: {len(self.indexes)}')

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return len(self.indexes) // self.batch_size

    def get_len(self) -> int:
        """Return total number of samples."""
        return len(self.indexes)

    def on_epoch_end(self) -> None:
        """Shuffle indexes at the end of each epoch."""
        shuffle(self.indexes)

    def __shape__(self) -> Tuple:
        """Return shape of a sample batch for debugging."""
        data = self.__getitem__(0)
        if isinstance(data, tuple):
            return tuple(d.shape if hasattr(d, 'shape') else None for d in data)
        return data.shape

    def generate_weight_map(self, mask: np.ndarray) -> np.ndarray:
        """
        Generate a weight map based on distance transform.

        The weight map gives higher weights to regions near the mask boundary,
        which helps the registration focus on edges.

        Args:
            mask: Binary mask array

        Returns:
            Weight map with same shape as mask, values in [0, 1]
        """
        decay_rate = 0.1
        seg_map = np.copy(mask)
        seg_map[seg_map > 0] = 1
        seg_map_inv = ~seg_map.astype(bool)

        # Compute distance transform (distance to nearest foreground pixel)
        distance_map = edt(seg_map_inv)

        # Convert distance to weight (exponential decay)
        weight_map = np.exp(-decay_rate * distance_map)

        return weight_map.astype(np.float32)

    def preprocess(self, im: np.ndarray) -> np.ndarray:
        """Normalize image intensity."""
        return normalize(im)

    def __getitem__(self, idx: int) -> Tuple[Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]]:
        """
        Get a batch of data for training.

        Args:
            idx: Batch index

        Returns:
            Tuple of (inputs, targets) where:
                inputs: (template, moving_images, moving_masks, [weight_map])
                targets: (warped_template, template_masks, moving_images, ...)
        """
        with threading.Lock():
            # Get indices for this batch
            start_idx = idx * self.batch_size
            end_idx = (idx + 1) * self.batch_size
            batch_indexes = [self.indexes[i] for i in range(start_idx, end_idx)]

            # Extract template images (fixed reference)
            # Shape: (batch_size, H, W) -> add channel dimension
            template = np.stack([
                normalize(self.pairs[index[0]][self.contrast_template[0]]['img'][index[1], :, :, index[3]])
                for index in batch_indexes
            ], axis=0)[..., np.newaxis]

            # Extract moving images (to be registered)
            # Shape: (batch_size, num_moving, H, W) -> add channel dimension
            im = np.stack([
                np.stack([
                    normalize(self.pairs[index[0]][c]['img'][index[1], ..., index[2][i]])
                    for i, c in enumerate(self.contrast_moving)
                ], axis=0)
                for index in batch_indexes
            ], axis=0)[..., np.newaxis]

            # Tile template to match number of moving contrasts
            # Shape: (batch_size, num_moving, H, W, 1)
            template_tiled = np.tile(template[:, np.newaxis], (1, len(self.contrast_moving), 1, 1, 1))

            # Extract moving masks
            # Shape: (batch_size, num_moving, H, W, 1)
            lbl_im = np.stack([
                np.stack([
                    self.pairs[index[0]][c]['mask'][index[1]]
                    for i, c in enumerate(self.contrast_moving)
                ], axis=0)
                for index in batch_indexes
            ], axis=0)[..., np.newaxis]

            # Extract template masks
            # Shape: (batch_size, H, W, 1)
            lbl_template = np.stack([
                self.pairs[index[0]][self.contrast_template[0]]['mask'][index[1]]
                for index in batch_indexes
            ], axis=0)[..., np.newaxis]

            # Tile template masks to match number of moving contrasts
            # Shape: (batch_size, num_moving, H, W, 1)
            lbl_template_tiled = np.tile(lbl_template[:, np.newaxis], (1, len(self.contrast_moving), 1, 1, 1))

            # Handle weighted version
            if self.weighted:
                # Create weight map from moving masks
                # Check if any foreground in any moving contrast
                lbl_weights = np.any(lbl_im, axis=1, keepdims=True)
                weight_map = self.generate_weight_map(lbl_weights).astype(np.float32)

                # Tile weight map to match number of moving contrasts
                weight_map_tiled = np.tile(weight_map, (1, len(self.contrast_moving), 1, 1, 1))

                # Apply weighting to template
                weighted_template = template_tiled * weight_map_tiled

                if self.affine:
                    # Reshape for affine network (flatten batch and moving dimensions)
                    weighted_template_flat = np.reshape(
                        weighted_template,
                        (-1, self.img_size_x, self.img_size_y, 1)
                    )
                    lbl_template_flat = np.reshape(
                        lbl_template_tiled,
                        (-1, self.img_size_x, self.img_size_y, 1)
                    )

                    # Inputs: template, moving images, moving masks, weight map
                    inputs = (
                        tf.convert_to_tensor(template_tiled),
                        tf.convert_to_tensor(im),
                        tf.convert_to_tensor(lbl_im),
                        tf.convert_to_tensor(weight_map_tiled)
                    )

                    # Targets: weighted template, template masks, moving images
                    targets = (
                        tf.convert_to_tensor(weighted_template_flat),
                        tf.convert_to_tensor(lbl_template_flat),
                        tf.convert_to_tensor(im)
                    )

                    return inputs, targets
                else:
                    # For non-affine (deformable) registration
                    inputs = (
                        tf.convert_to_tensor(template_tiled),
                        tf.convert_to_tensor(im),
                        tf.convert_to_tensor(lbl_im),
                        tf.convert_to_tensor(weight_map_tiled)
                    )

                    targets = (
                        tf.convert_to_tensor(weighted_template),
                        tf.convert_to_tensor(lbl_template_tiled),
                        tf.convert_to_tensor(im),
                        tf.convert_to_tensor(im)  # Duplicate for additional loss
                    )

                    return inputs, targets

            else:
                # Unweighted version
                if self.affine:
                    # Reshape for affine network
                    reshaped_template = np.reshape(
                        template_tiled,
                        (-1, self.img_size_x, self.img_size_y, 1)
                    )
                    lbl_template_flat = np.reshape(
                        lbl_template_tiled,
                        (-1, self.img_size_x, self.img_size_y, 1)
                    )

                    inputs = (
                        tf.convert_to_tensor(template_tiled),
                        tf.convert_to_tensor(im),
                        tf.convert_to_tensor(lbl_im)
                    )

                    targets = (
                        tf.convert_to_tensor(reshaped_template),
                        tf.convert_to_tensor(lbl_template_flat),
                        tf.convert_to_tensor(im)
                    )

                    return inputs, targets

                else:
                    # For non-affine (deformable) registration
                    inputs = (
                        tf.convert_to_tensor(template_tiled),
                        tf.convert_to_tensor(im),
                        tf.convert_to_tensor(lbl_im)
                    )

                    targets = (
                        tf.convert_to_tensor(template_tiled),
                        tf.convert_to_tensor(lbl_template_tiled),
                        tf.convert_to_tensor(im),
                        tf.convert_to_tensor(im)  # Duplicate for additional loss
                    )

                    return inputs, targets