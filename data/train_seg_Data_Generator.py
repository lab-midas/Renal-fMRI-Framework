"""
Data loader for segmentation fine-tuning.

This module provides a custom Keras Sequence data generator that loads
multi-contrast MRI images and corresponding segmentation masks for
kidney segmentation tasks.

Supports two tasks:
    - 'volume': 3-class segmentation (background, right kidney, left kidney)
    - 'cortex': 5-class segmentation (background, right cortex, left cortex,
                 right medulla, left medulla)
"""

import glob
import os
import random
import threading
from random import shuffle
from typing import List, Tuple, Optional, Any

import albumentations as A
import nibabel as nib
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from .data_utils import normalize, transpose_array


class DataLoader(Sequence):
    """
    Custom data generator for segmentation fine-tuning.

    This class loads multi-contrast MRI images and corresponding segmentation
    masks, applies data augmentation (for training), and generates batches
    for model training/validation.

    Attributes:
        batch_size (int): Number of samples per batch
        num_channels (int): Number of image channels
        cfg: Configuration object
        train_flag (bool): Whether this is training or validation loader
        contrast (list): List of MRI contrasts to load
        img_size_x, img_size_y (int): Image dimensions
        input_img (list): List of subject IDs
        aug (albumentations.Compose): Data augmentation pipeline
        data_dir (str): Directory containing image data
        img_labels_list (list): List of (image, label) tuples
    """

    def __init__(self, cfg: Any, debug: bool, train_flag: bool = True):
        """
        Initialize the data loader.

        Args:
            cfg: Configuration object
            debug: If True, load limited data for debugging
            train_flag: If True, load training data with augmentation;
                       else load validation data without augmentation
        """
        self.batch_size = cfg.batch_size
        self.num_channels = cfg.num_channels
        self.cfg = cfg
        self.train_flag = train_flag
        self.contrast = cfg.contrast
        self.img_size_x = cfg.img_size_x
        self.img_size_y = cfg.img_size_x
        self.data_dir = cfg.data_dir

        # Load subject list
        if train_flag:
            print('Initializing training dataloader')
            subject_file = cfg.train_sub
            # For debugging, limit to first subject
            self.input_img = np.load(subject_file, allow_pickle=True).tolist()[:1] if debug else np.load(subject_file,
                                                                                                         allow_pickle=True).tolist()

            # Data augmentation pipeline for training
            self.aug = A.OneOf([
                A.NoOp(p=0.1),  # 10% chance of no augmentation
                A.Compose([
                    A.ElasticTransform(alpha=20, sigma=20, p=0.3),  # Elastic deformation
                    A.RandomBrightnessContrast(p=0.3),  # Intensity augmentation
                    A.OneOf([
                        A.Affine(translate_percent=(0, 0.15), p=0.5),  # Translation
                        A.Affine(scale=(0.9, 1), p=0.5),  # Scaling
                    ], p=0.5),
                    A.RandomResizedCrop(
                        size=(self.img_size_x, self.img_size_y),
                        scale=(0.8, 1),
                        p=0.3
                    ),  # Random crop and resize
                ], p=0.9)
            ], p=1)
        else:
            print('Initializing validation dataloader')
            subject_file = cfg.val_sub
            self.input_img = np.load(subject_file, allow_pickle=True).tolist()[:1] if debug else np.load(subject_file,
                                                                                                         allow_pickle=True).tolist()

        # Store all (image, label) pairs
        self.img_labels_list = []

        # Load data based on task
        if cfg.task == 'volume':
            self.load_volumes()
        elif cfg.task == 'cortex':
            self.load_cortex()
        else:
            raise NotImplementedError(f"Task {cfg.task} not implemented")

        # Shuffle data
        shuffle(self.img_labels_list)

        # Calculate dataset size
        self.len_of_data = len(self.img_labels_list)
        self.num_samples = self.len_of_data // 1
        print(f'Total samples: {self.num_samples}')

    def load_cortex(self) -> None:
        """
        Load data for 5-class cortex/medulla segmentation.

        Loads images and corresponding masks for right/left cortex and medulla.
        Creates 5-class labels: [background, right_cortex, left_cortex,
                                 right_medulla, left_medulla]
        """
        for contrast in self.contrast:
            for sub in self.input_img:
                # Find all image and label files
                path_img = glob.glob(os.path.join(self.data_dir, sub, contrast, 'imagesTr/*'))
                path_right_vol = glob.glob(os.path.join(self.data_dir, sub, contrast, 'labelsTr/*right*volume*'))
                path_left_vol = glob.glob(os.path.join(self.data_dir, sub, contrast, 'labelsTr/*left*volume*'))
                path_right_cortex = glob.glob(os.path.join(self.data_dir, sub, contrast, 'labelsTr/*right*cortex*'))
                path_left_cortex = glob.glob(os.path.join(self.data_dir, sub, contrast, 'labelsTr/*left*cortex*'))

                # Check if all required files exist
                if all([path_img, path_right_vol, path_left_vol, path_right_cortex, path_left_cortex]):
                    # Create pairs of (image_path, right_vol_path, left_vol_path, right_cortex_path, left_cortex_path)
                    paths = [((x, path_right_vol[0], path_left_vol[0], path_right_cortex[0], path_left_cortex[0]))
                             for x in path_img]

                    for (p_im, p_vol_r, p_vol_l, p_cor_r, p_cor_l) in paths:
                        # Load and preprocess image
                        im = np.flip(np.rot90(nib.load(p_im).get_fdata(), -1), 1).astype('float32')
                        im = transpose_array(im)  # Reorder dimensions to (slices, H, W)

                        # Load and preprocess labels
                        right_vol_lbl = np.flip(np.rot90(nib.load(p_vol_r).get_fdata(), -1), 1).astype('float32')
                        left_vol_lbl = np.flip(np.rot90(nib.load(p_vol_l).get_fdata(), -1), 1).astype('float32')
                        right_cor_lbl = np.flip(np.rot90(nib.load(p_cor_r).get_fdata(), -1), 1).astype('float32')
                        left_cor_lbl = np.flip(np.rot90(nib.load(p_cor_l).get_fdata(), -1), 1).astype('float32')

                        # Transpose labels
                        right_vol_lbl = transpose_array(right_vol_lbl)
                        left_vol_lbl = transpose_array(left_vol_lbl)
                        right_cor_lbl = transpose_array(right_cor_lbl)
                        left_cor_lbl = transpose_array(left_cor_lbl)

                        # Calculate medulla masks (volume - cortex)
                        right_medulla_lbl = right_vol_lbl - right_cor_lbl
                        left_medulla_lbl = left_vol_lbl - left_cor_lbl

                        # Create background mask
                        background = np.ones_like(left_cor_lbl)
                        background = background - right_cor_lbl - left_cor_lbl - right_medulla_lbl - left_medulla_lbl

                        # Stack into 5-class label: [bg, R_cortex, L_cortex, R_medulla, L_medulla]
                        lbl = np.stack([background, right_cor_lbl, left_cor_lbl, right_medulla_lbl, left_medulla_lbl],
                                       axis=-1)

                        # Filter out slices without any kidney tissue
                        non_zero_fix = np.any(right_vol_lbl != 0, (1, 2)).tolist()
                        non_zero_mov = np.any(left_vol_lbl != 0, (1, 2)).tolist()
                        non_zero = [x or y for x, y in zip(non_zero_mov, non_zero_fix)]
                        non_zero = np.asarray(non_zero)

                        # Keep only slices with kidney tissue
                        im = im[non_zero]
                        lbl = lbl[non_zero]

                        # Add each slice to dataset
                        for b in range(im.shape[0]):
                            if im.ndim == 3:  # Single channel
                                self.img_labels_list.append((im[b], lbl[b]))
                            if im.ndim == 4:  # Multiple channels (e.g., multi-echo)
                                for b2 in range(im.shape[-1]):
                                    self.img_labels_list.append((im[b][..., b2], lbl[b]))

    def load_volumes(self) -> None:
        """
        Load data for 3-class whole kidney segmentation.

        Loads images and corresponding masks for right/left kidney.
        Creates 3-class labels: [background, right_kidney, left_kidney]
        """
        for contrast in self.contrast:
            for sub in self.input_img:
                # Find all image and label files
                path_img = glob.glob(os.path.join(self.data_dir, sub, contrast, 'imagesTr/*'))
                path_right_lbl = glob.glob(os.path.join(self.data_dir, sub, contrast, 'labelsTr/*right*volume*'))
                path_left_lbl = glob.glob(os.path.join(self.data_dir, sub, contrast, 'labelsTr/*left*volume*'))

                # Check if all required files exist
                if path_img and path_right_lbl and path_left_lbl:
                    # Create pairs of (image_path, right_label_path, left_label_path)
                    paths = [((x, path_right_lbl[0], path_left_lbl[0])) for x in path_img]

                    for (p_im, l_r, l_l) in paths:
                        # Load and preprocess image
                        im = np.flip(np.rot90(nib.load(p_im).get_fdata(), -1), 1).astype('float32')
                        im = transpose_array(im)  # Reorder dimensions to (slices, H, W)

                        # Load and preprocess labels
                        right_lbl = np.flip(np.rot90(nib.load(l_r).get_fdata(), -1), 1).astype('float32')
                        left_lbl = np.flip(np.rot90(nib.load(l_l).get_fdata(), -1), 1).astype('float32')
                        right_lbl = transpose_array(right_lbl)
                        left_lbl = transpose_array(left_lbl)

                        # Filter out slices without any kidney tissue
                        non_zero_fix = np.any(right_lbl != 0, (1, 2)).tolist()
                        non_zero_mov = np.any(left_lbl != 0, (1, 2)).tolist()
                        non_zero = [x or y for x, y in zip(non_zero_mov, non_zero_fix)]
                        non_zero = np.asarray(non_zero)

                        # Keep only slices with kidney tissue
                        im = im[non_zero]
                        right_lbl = right_lbl[non_zero]
                        left_lbl = left_lbl[non_zero]

                        # Create background mask
                        background = np.ones_like(left_lbl)
                        background = background - right_lbl - left_lbl

                        # Stack into 3-class label: [bg, right_kidney, left_kidney]
                        lbl = np.stack([background, right_lbl, left_lbl], axis=-1)

                        # Add each slice to dataset
                        for b in range(im.shape[0]):
                            if im.ndim == 3:  # Single channel
                                self.img_labels_list.append((im[b], lbl[b]))
                            if im.ndim == 4:  # Multiple channels
                                for b2 in range(im.shape[-1]):
                                    self.img_labels_list.append((im[b][..., b2], lbl[b]))

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return len(self.img_labels_list) // self.batch_size

    def get_len(self) -> int:
        """Return total number of samples."""
        return len(self.img_labels_list)

    def on_epoch_end(self) -> None:
        """Shuffle data at the end of each epoch."""
        shuffle(self.img_labels_list)

    def __shape__(self) -> Tuple:
        """Return shape of a sample batch."""
        data = self.__getitem__(0)
        return data[0].shape

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Normalize image intensity."""
        return normalize(img)

    def __getitem__(self, idx: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get a batch of data.

        Args:
            idx: Batch index

        Returns:
            Tuple of (images, labels) as TensorFlow tensors
        """
        with threading.Lock():
            # Get indices for this batch
            start_idx = idx * self.batch_size
            end_idx = (idx + 1) * self.batch_size
            img_lbl_pair = [self.img_labels_list[i] for i in range(start_idx, end_idx)]

            # Stack images and labels
            im = np.stack([self.preprocess(data[0]) for data in img_lbl_pair])[..., None]
            lbl = np.stack([data[1] for data in img_lbl_pair])

            # Apply data augmentation for training
            if self.train_flag:
                random.seed(7)  # Fixed seed for reproducibility
                augmented = [self.aug(image=im[i], mask=lbl[i]) for i in range(self.batch_size)]
                im = np.stack([augmented[i]['image'] for i in range(self.batch_size)], axis=0)
                lbl = np.stack([augmented[i]['mask'] for i in range(self.batch_size)], axis=0)

            # Convert to TensorFlow tensors
            im = tf.identity(im)
            lbl = tf.identity(lbl)

            return im, lbl