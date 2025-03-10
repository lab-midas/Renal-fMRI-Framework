import glob
import os
import random
import threading
from random import shuffle

import albumentations as A
import nibabel as nib
import numpy as np
import tensorflow as tf
import tensorflow.keras
from scipy.ndimage import affine_transform


class DataLoader(tensorflow.keras.utils.Sequence):
    def __init__(self, cfg, debug, train_flag=True):
        self.batch_size = cfg.batch_size
        self.num_channels = cfg.num_channels
        self.cfg = cfg
        self.train_flag = train_flag
        self.contrast = cfg.contrast
        self.img_size_x = cfg.img_size_x
        self.img_size_y = cfg.img_size_x
        if train_flag:
            print('Initializing training dataloader')
            self.input_img = np.load(cfg.train_sub, allow_pickle=True).tolist()
            self.aug = A.OneOf([A.NoOp(p=0.1),
                                A.Compose([A.ElasticTransform(alpha=20, sigma=20, p=0.3),
                                           A.RandomBrightnessContrast(p=0.3),
                                           A.OneOf([
                                               A.Affine(translate_percent=(0, 0.15), p=0.5),
                                               A.Affine(scale=(0.9, 1), p=0.5), ], p=0.5),
                                           A.RandomResizedCrop(size=(self.img_size_x,self.img_size_y), scale=(0.8, 1),
                                                               p=0.3)], p=0.9)], p=1)
        else:
            print('Initializing validation dataloader')
            self.input_img = np.load(cfg.val_sub, allow_pickle=True).tolist()
        #if debug:
        #    self.input_img = self.input_img[:2]
        self.data_dir = cfg.data_dir
        self.img_labels_list = []
        if cfg.task == 'volume':
            self.load_volumes()
        elif cfg.task == 'cortex':
            self.load_cortex()
        else:
            raise NotImplementedError
        shuffle(self.img_labels_list)
        self.len_of_data = len(self.img_labels_list)
        self.num_samples = self.len_of_data // 1  # use it to control size of sampels per epoch
        print('Total samples :', self.num_samples)

    def load_cortex(self):
        for contrast in self.contrast:
            for sub in self.input_img:
                path_img = glob.glob(os.path.join(self.data_dir, sub, contrast, f'imagesTr/*'))
                path_right_vol = glob.glob(os.path.join(self.data_dir, sub, contrast, f'labelsTr/*right*volume*'))
                path_left_vol = glob.glob(os.path.join(self.data_dir, sub, contrast, f'labelsTr/*left*volume*'))
                path_right_cortex = glob.glob(os.path.join(self.data_dir, sub, contrast, f'labelsTr/*right*cortex*'))
                path_left_cortex = glob.glob(os.path.join(self.data_dir, sub, contrast, f'labelsTr/*left*cortex*'))
                #print(path_left_vol, path_right_vol, path_img)
                if path_img and path_right_vol and path_left_vol and path_right_cortex and path_left_cortex:
                    paths = [((x, path_right_vol[0],path_left_vol[0], path_right_cortex[0], path_left_cortex[0])) for x in path_img]
                    for ((p_im, p_vol_r, p_vol_l, p_cor_r, p_cor_l)) in paths:
                        # load images
                        im = np.flip(np.rot90(nib.load(p_im).get_fdata(), -1),1).astype('float32')
                        im = transpose_array(im)

                        # load labels
                        #print(l_r, l_m)
                        right_vol_lbl = np.flip(np.rot90(nib.load(p_vol_r).get_fdata(), -1),1).astype('float32')
                        left_vol_lbl = np.flip(np.rot90(nib.load(p_vol_l).get_fdata(), -1),1).astype('float32')
                        right_cor_lbl = np.flip(np.rot90(nib.load(p_cor_r).get_fdata(), -1),1).astype('float32')
                        left_cor_lbl = np.flip(np.rot90(nib.load(p_cor_l).get_fdata(), -1),1).astype('float32')

                        right_vol_lbl = transpose_array(right_vol_lbl)
                        left_vol_lbl = transpose_array(left_vol_lbl)
                        right_cor_lbl = transpose_array(right_cor_lbl)
                        left_cor_lbl = transpose_array(left_cor_lbl)

                        right_medulla_lbl = right_vol_lbl - right_cor_lbl
                        left_medulla_lbl = left_vol_lbl - left_cor_lbl

                        background = np.ones_like(left_cor_lbl)
                        background = background - right_cor_lbl - left_cor_lbl - right_medulla_lbl - left_medulla_lbl
                        lbl = np.stack([background, right_cor_lbl, left_cor_lbl, right_medulla_lbl, left_medulla_lbl], axis=-1)

                        # sort slices that don't contain kidneys
                        non_zero_fix = np.any(right_vol_lbl != 0, (1, 2)).tolist()
                        non_zero_mov = np.any(left_vol_lbl != 0, (1, 2)).tolist()
                        non_zero = [x or y for x, y in zip(non_zero_mov, non_zero_fix)]
                        non_zero = np.asarray(non_zero)
                        #print(non_zero)

                        im = im[non_zero]
                        lbl = lbl[non_zero]

                        for b in range(im.shape[0]):
                            if im.ndim == 3:
                                self.img_labels_list.append((im[b], lbl[b]))
                            if im.ndim == 4:
                                for b2 in range(im.shape[-1]):
                                    self.img_labels_list.append((im[b][...,b2], lbl[b]))

    def load_volumes(self):
        for contrast in self.contrast:
            for sub in self.input_img:
                path_img = glob.glob(os.path.join(self.data_dir, sub, contrast, f'imagesTr/*'))
                path_right_lbl = glob.glob(os.path.join(self.data_dir, sub, contrast, f'labelsTr/*right*volume*'))
                path_left_lbl = glob.glob(os.path.join(self.data_dir, sub, contrast, f'labelsTr/*left*volume*'))
                #print(path_left_lbl, path_right_lbl, path_img)
                if path_img and path_right_lbl and path_left_lbl:
                    paths = [((x, path_right_lbl[0],path_left_lbl[0])) for x in path_img]
                    for ((p_im, l_r, l_l)) in paths:
                        # load images
                        im = np.flip(np.rot90(nib.load(p_im).get_fdata(), -1),1).astype('float32')
                        im = transpose_array(im)

                        # load labels
                        #print(l_r, l_m)
                        right_lbl = np.flip(np.rot90(nib.load(l_r).get_fdata(), -1),1).astype('float32')
                        left_lbl = np.flip(np.rot90(nib.load(l_l).get_fdata(), -1),1).astype('float32')
                        right_lbl = transpose_array(right_lbl)
                        left_lbl = transpose_array(left_lbl)

                        # sort slices that don't contain kidneys
                        non_zero_fix = np.any(right_lbl != 0, (1, 2)).tolist()
                        non_zero_mov = np.any(left_lbl != 0, (1, 2)).tolist()
                        non_zero = [x or y for x, y in zip(non_zero_mov, non_zero_fix)]
                        non_zero = np.asarray(non_zero)
                        #print(non_zero)

                        im = im[non_zero]
                        right_lbl = right_lbl[non_zero]
                        left_lbl = left_lbl[non_zero]

                        background = np.ones_like(left_lbl)
                        background = background - right_lbl - left_lbl
                        lbl = np.stack([background, right_lbl, left_lbl], axis=-1)

                        for b in range(im.shape[0]):
                            if im.ndim == 3:
                                self.img_labels_list.append((im[b], lbl[b]))
                            if im.ndim == 4:
                                for b2 in range(im.shape[-1]):
                                    self.img_labels_list.append((im[b][...,b2], lbl[b]))



    def __len__(self):
        return len(self.img_labels_list) // self.batch_size

    def get_len(self):
        return len(self.img_labels_list)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        shuffle(self.img_labels_list)


    def __shape__(self):
        data = self.__getitem__(0)
        return data.shape

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        with threading.Lock():
            # Generate indices of the batch
            img_lbl_pair = [self.img_labels_list[i] for i in range(idx * self.batch_size, (idx + 1) * self.batch_size)]
            im = np.stack([self.preprocess(data[0]) for data in img_lbl_pair])[...,None]
            lbl = np.stack([data[1] for data in img_lbl_pair])
            if self.train_flag:
                random.seed(7)
                augmented = [self.aug(image=im[i], mask=lbl[i]) for i in range(self.batch_size)]
                im = np.stack([augmented[i]['image'] for i in range(self.batch_size)], axis=0)
                lbl = np.stack([augmented[i]['mask'] for i in range(self.batch_size)], axis=0)
            im = tf.identity(im)
            lbl = tf.identity(lbl)

            return im, lbl

    def preprocess(self, img):
        return normalize(img)



class DataLoader3D(tensorflow.keras.utils.Sequence):
    def __init__(self, cfg, debug, train_flag=True):
        self.batch_size = cfg.batch_size
        self.num_channels = cfg.num_channels
        self.cfg = cfg
        self.train_flag = train_flag
        self.contrast = cfg.contrast
        self.img_size_x = cfg.img_size_x
        self.img_size_y = cfg.img_size_y
        self.img_size_z = cfg.img_size_z
        if train_flag:
            print('Initializing training dataloader')
            self.input_img = np.load(cfg.train_sub, allow_pickle=True).tolist()
            self.aug = A.OneOf([A.NoOp(p=0.2),
                                A.Compose([A.ElasticTransform(alpha=20, sigma=20, p=0.2),
                                           A.RandomBrightnessContrast(p=0.2),
                                           A.Affine(translate_percent=(0, 0.15), p=0.6),], p=0.8)], p=1)
        else:
            print('Initializing validation dataloader')
            self.input_img = np.load(cfg.val_sub, allow_pickle=True).tolist()
            self.aug = False
        if debug:
            self.input_img = self.input_img[:2]
        self.data_dir = cfg.data_dir
        self.img_labels_list = []
        if cfg.task == 'volume':
            self.load_volumes()
        elif cfg.task == 'cortex':
            self.load_cortex()
        else:
            raise NotImplementedError
        shuffle(self.img_labels_list)
        self.len_of_data = len(self.img_labels_list)
        self.num_samples = self.len_of_data // 1  # use it to control size of sampels per epoch
        print('Total samples :', self.num_samples)

    def load_cortex(self):
        for contrast in self.contrast:
            for sub in self.input_img:
                path_img = glob.glob(os.path.join(self.data_dir, sub, contrast, f'imagesTr/*'))
                path_right_vol = glob.glob(os.path.join(self.data_dir, sub, contrast, f'labelsTr/*right*volume*'))
                path_left_vol = glob.glob(os.path.join(self.data_dir, sub, contrast, f'labelsTr/*left*volume*'))
                path_right_cortex = glob.glob(os.path.join(self.data_dir, sub, contrast, f'labelsTr/*right*cortex*'))
                path_left_cortex = glob.glob(os.path.join(self.data_dir, sub, contrast, f'labelsTr/*left*cortex*'))
                #print(path_left_vol, path_right_vol, path_img)
                if path_img and path_right_vol and path_left_vol and path_right_cortex and path_left_cortex:
                    paths = [((x, path_right_vol[0],path_left_vol[0], path_right_cortex[0], path_left_cortex[0])) for x in path_img]
                    for ((p_im, p_vol_r, p_vol_l, p_cor_r, p_cor_l)) in paths:
                        # load images
                        im = np.flip(np.rot90(nib.load(p_im).get_fdata(), -1),1).astype('float32')
                        im = transpose_array(im)

                        # load labels
                        #print(l_r, l_m)
                        right_vol_lbl = np.flip(np.rot90(nib.load(p_vol_r).get_fdata(), -1),1).astype('float32')
                        left_vol_lbl = np.flip(np.rot90(nib.load(p_vol_l).get_fdata(), -1),1).astype('float32')
                        right_cor_lbl = np.flip(np.rot90(nib.load(p_cor_r).get_fdata(), -1),1).astype('float32')
                        left_cor_lbl = np.flip(np.rot90(nib.load(p_cor_l).get_fdata(), -1),1).astype('float32')

                        right_vol_lbl = transpose_array(right_vol_lbl)
                        left_vol_lbl = transpose_array(left_vol_lbl)
                        right_cor_lbl = transpose_array(right_cor_lbl)
                        left_cor_lbl = transpose_array(left_cor_lbl)

                        right_medulla_lbl = right_vol_lbl - right_cor_lbl
                        left_medulla_lbl = left_vol_lbl - left_cor_lbl

                        background = np.ones_like(left_cor_lbl)
                        background = background - right_cor_lbl - left_cor_lbl - right_medulla_lbl - left_medulla_lbl
                        lbl = np.stack([background, right_cor_lbl, left_cor_lbl, right_medulla_lbl, left_medulla_lbl], axis=-1)

                        # sort slices that don't contain kidneys
                        #non_zero_fix = np.any(right_vol_lbl != 0, (1, 2)).tolist()
                        #non_zero_mov = np.any(left_vol_lbl != 0, (1, 2)).tolist()
                        #non_zero = [x or y for x, y in zip(non_zero_mov, non_zero_fix)]
                        #non_zero = np.asarray(non_zero)
                        #print(non_zero)

                        #im = im[non_zero]
                        #lbl = lbl[non_zero]

                        if im.ndim == 3:
                            im = np.transpose(im, (1,2,0))
                            lbl = np.transpose(lbl, (1,2,0,3))
                            self.img_labels_list.append((im, lbl))
                        if im.ndim == 4:
                            im = np.transpose(im, (1,2,0,3))
                            lbl = np.transpose(lbl, (1,2,0,3))
                            for b2 in range(im.shape[-1]):
                                self.img_labels_list.append((im[...,b2], lbl))

    def load_volumes(self):
        for contrast in self.contrast:
            for sub in self.input_img:
                path_img = glob.glob(os.path.join(self.data_dir, sub, contrast, f'imagesTr/*'))
                path_right_lbl = glob.glob(os.path.join(self.data_dir, sub, contrast, f'labelsTr/*right*volume*'))
                path_left_lbl = glob.glob(os.path.join(self.data_dir, sub, contrast, f'labelsTr/*left*volume*'))
                #print(path_left_lbl, path_right_lbl, path_img)
                if path_img and path_right_lbl and path_left_lbl:
                    paths = [((x, path_right_lbl[0],path_left_lbl[0])) for x in path_img]
                    for ((p_im, l_r, l_l)) in paths:
                        # load images
                        im = np.flip(np.rot90(nib.load(p_im).get_fdata(), -1),1).astype('float32')
                        im = transpose_array(im)

                        # load labels
                        #print(l_r, l_m)
                        right_lbl = np.flip(np.rot90(nib.load(l_r).get_fdata(), -1),1).astype('float32')
                        left_lbl = np.flip(np.rot90(nib.load(l_l).get_fdata(), -1),1).astype('float32')
                        right_lbl = transpose_array(right_lbl)
                        left_lbl = transpose_array(left_lbl)

                        # sort slices that don't contain kidneys
                        #non_zero_fix = np.any(right_lbl != 0, (1, 2)).tolist()
                        #non_zero_mov = np.any(left_lbl != 0, (1, 2)).tolist()
                        #non_zero = [x or y for x, y in zip(non_zero_mov, non_zero_fix)]
                        #non_zero = np.asarray(non_zero)
                        #print(non_zero)

                        #im = im[non_zero]
                        #right_lbl = right_lbl[non_zero]
                        #left_lbl = left_lbl[non_zero]

                        background = np.ones_like(left_lbl)
                        background = background - right_lbl - left_lbl
                        lbl = np.stack([background, right_lbl, left_lbl], axis=-1)



                        if im.ndim == 3:
                            im = np.transpose(im, (1, 2, 0))
                            lbl = np.transpose(lbl, (1, 2, 0, 3))
                            #print(im.shape, lbl.shape)
                            self.img_labels_list.append((im, lbl))
                        if im.ndim == 4:
                            im = np.transpose(im, (1, 2, 0, 3))
                            lbl = np.transpose(lbl, (1, 2, 0, 3))
                            for b2 in range(im.shape[-1]):
                                #print(im[..., b2].shape, lbl.shape)
                                self.img_labels_list.append((im[..., b2], lbl))



    def __len__(self):
        return len(self.img_labels_list) // self.batch_size

    def get_len(self):
        return len(self.img_labels_list)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        shuffle(self.img_labels_list)


    def __shape__(self):
        data = self.__getitem__(0)
        return data.shape

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        with threading.Lock():
            # Generate indices of the batch
            img_lbl_pair = [self.img_labels_list[i] for i in range(idx * self.batch_size, (idx + 1) * self.batch_size)]
            #for im in  img_lbl_pair:
            #    print(im[0].shape)
            im = np.stack([self.preprocess(data[0]) for data in img_lbl_pair])[...,None]
            lbl = np.stack([data[1] for data in img_lbl_pair])
            if self.aug:
                random.seed(7)
                im, lbl = self.apply_3d_augmentation(im, lbl)
            im = tf.identity(im)
            lbl = tf.identity(lbl)

            return im, lbl

    def preprocess(self, img):
        return np.stack([normalize(img[:,:,i]) for i in range(self.img_size_z)], axis=-1)

    def apply_3d_augmentation(self, im, lbl):
        for i in range(self.img_size_z):
            for b in range(self.batch_size):
                augmented = self.aug(image=im[b,:,:,i], mask=lbl[b,:,:,i])
                im[b,:,:,i] = augmented['image']
                lbl[b,:,:,i] = augmented['mask']
        return im, lbl


def translate_3d(image, shift):
        """
        Translates a 3D volume (image) in x, y, and z directions.

        Args:
        - image: 3D Tensor or NumPy array of shape (D, H, W).
        - shift: List of 3 values [shift_z, shift_y, shift_x].

        Returns:
        - Translated 3D image
        """
        shift_matrix = np.array(shift, dtype=np.float32)  # Convert shift values to float32
        translated_image = affine_transform(image.numpy(), matrix=np.eye(3), offset=shift_matrix, order=1)
        return tf.convert_to_tensor(translated_image, dtype=tf.float32)



def transpose_array(arr):
    if arr.ndim == 3:
        return np.transpose(arr, (2,0,1))
    elif arr.ndim == 4:
        return np.transpose(arr, (2, 0,1, 3))
    else:
        raise ValueError("Input array must be 3D or 4D.")


def normalize(img):
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:  # Avoid division by zero
        return (img - img_min) / (img_max - img_min)
    else:
        return np.zeros_like(img)


