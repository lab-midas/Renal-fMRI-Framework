import tensorflow.keras
import numpy as np
import threading
import tensorflow as tf
import os
from random import shuffle
import glob
import nibabel as nib
from scipy.ndimage import distance_transform_edt as edt
from tqdm import tqdm
import scipy.io as sio


class DataLoaderPairwise(tensorflow.keras.utils.Sequence):
    def __init__(self, cfg, debug, train_flag=True):
        self.batch_size = cfg.batch_size
        self.debug = debug
        self.num_channels = cfg.num_channels
        self.cfg = cfg
        self.type_mask = cfg.type_mask
        self.out_features = cfg.out_features
        self.affine = cfg.affine
        self.ft_training = cfg.ft_training
        self.weighted = cfg.weighted
        self.train_flag = train_flag
        self.size = (cfg.img_size_x, cfg.img_size_y)
        if train_flag:
            print('Initializing training dataloader')
            self.input_img = np.load(cfg.train_sub).tolist()
        else:
            print('Initializing validation dataloader')
            self.input_img = np.load(cfg.val_sub).tolist()
        if self.debug:
            self.input_img = self.input_img[2:3]
        self.data_dir = cfg.data_dir
        #self.ft_dir = cfg.ft_dir
        self.out_labels = cfg.out_labels
        self.contrast_ref = cfg.contrast_ref
        self.contrast_mov = cfg.contrast_mov
        self.pairs = []
        self.indexes = []
        self.load_img_ft() if self.ft_training else self.load_imgs()
        shuffle(self.indexes)
        self.len_of_data = len(self.indexes)
        self.num_samples = self.len_of_data // 1  # use it to control size of sampels per epoch
        print('Total samples :', self.num_samples)
        self.img_size_x = cfg.img_size_x
        self.img_size_y = cfg.img_size_x

    def load_img_ft(self):
        idx = 0
        for sub in tqdm(self.input_img):
            ref_data_path = os.path.join(self.data_dir, self.contrast_ref, f'Constraint_map_{sub}_20.mat')
            mov_data_path = os.path.join(self.data_dir, self.contrast_mov, f'Constraint_map_{sub}_20.mat')

            if os.path.exists(ref_data_path) and os.path.exists(mov_data_path):
                ref_data = sio.loadmat(ref_data_path)
                mov_data = sio.loadmat(mov_data_path)

                fix = transpose_array(ref_data['img'])
                mov = transpose_array(mov_data['img'])
                fix_lbl = transpose_array(ref_data['mask'])
                mov_lbl = transpose_array(mov_data['mask'])
                fix_param = transpose_array(ref_data['param'])
                mov_param = transpose_array(mov_data['param'])

                fix_lbl = np.where(fix_lbl != 0, 1, 0)
                mov_lbl = np.where(mov_lbl != 0, 1, 0)

                non_zero_fix = np.any(fix_lbl != 0, (1, 2)).tolist()
                non_zero_mov = np.any(mov_lbl != 0, (1, 2)).tolist()
                non_zero = [x and y for x, y in zip(non_zero_mov, non_zero_fix)]
                non_zero = np.asarray(non_zero)

                fix = fix[non_zero].astype('float32')
                mov = mov[non_zero].astype('float32')
                fix_lbl = fix_lbl[non_zero].astype('float32')
                mov_lbl = mov_lbl[non_zero].astype('float32')
                fix_param = fix_param[non_zero].astype('float32')
                mov_param = mov_param[non_zero].astype('float32')

                data = [fix, mov, fix_lbl, mov_lbl,fix_param, mov_param]

                self.pairs.append(data)

                for b1 in range(fix.shape[0]):
                    for b2 in range(fix.shape[-1]):
                        for b3 in range(mov.shape[-1]):
                            if fix[b1,...,b2].min() != fix[b1,...,b2].max() and mov[b1,..., b3].min() != mov[b1,..., b3].max():
                                self.indexes.append((idx, b1, b2, b3))
                idx += 1





            else:
                print(f'either ref and mov data for {sub} does not exist')

    def load_imgs(self):
        idx = 0
        for sub in tqdm(self.input_img):
            path_ref_img = glob.glob(os.path.join(self.data_dir, sub, self.contrast_ref, f'imagesTr/*'))
            path_mov_img = glob.glob(os.path.join(self.data_dir, sub, self.contrast_mov, f'imagesTr/*'))
            path_ref_lbl = glob.glob(os.path.join(self.data_dir, sub, self.contrast_ref, f'labelsTr/*volume*'))
            path_mov_lbl = glob.glob(os.path.join(self.data_dir, sub, self.contrast_mov, f'labelsTr/*volume*'))
            if path_ref_img and path_mov_img and path_ref_lbl and path_mov_lbl:
                paths = [((x, path_ref_lbl), (y,path_mov_lbl)) for x in path_ref_img for y in path_mov_img]
                for ((p_r, l_r), (p_m, l_m)) in paths:
                    fix = np.flip(np.rot90(nib.load(p_r).get_fdata(), -1), 1).astype('float32')
                    mov = np.flip(np.rot90(nib.load(p_m).get_fdata(), -1), 1).astype('float32')
                    fix = transpose_array(fix)
                    mov = transpose_array(mov)

                    # load labels
                    fix_lbl = np.amax(
                        np.stack([np.flip(np.rot90(nib.load(p).get_fdata().astype('float32'), -1), 1) for p in l_r],
                                 0), 0)
                    mov_lbl = np.amax(
                        np.stack([np.flip(np.rot90(nib.load(p).get_fdata().astype('float32'), -1), 1) for p in l_m],
                                 0), 0)
                    fix_lbl = transpose_array(fix_lbl)
                    mov_lbl = transpose_array(mov_lbl)

                    # sort slices that don't contain kidneys
                    non_zero_fix = np.any(fix_lbl != 0, (1, 2)).tolist()
                    non_zero_mov = np.any(mov_lbl != 0, (1, 2)).tolist()
                    non_zero = [x and y for x, y in zip(non_zero_mov, non_zero_fix)]

                    non_zero = np.asarray(non_zero)

                    fix_lbl = fix_lbl[non_zero]
                    mov_lbl = mov_lbl[non_zero]
                    fix = fix[non_zero]
                    mov = mov[non_zero]

                    fix = fix[...,None] if fix.ndim == 3 else fix
                    mov = mov[...,None] if mov.ndim == 3 else mov

                    self.pairs.append([fix, mov, fix_lbl, mov_lbl])

                    for b1 in range(fix.shape[0]):
                        for b2 in range(fix.shape[-1]):
                            for b3 in range(mov.shape[-1]):
                                if fix[b1, ..., b2].min() != fix[b1, ..., b2].max() and mov[b1, ..., b3].min() != mov[
                                    b1, ..., b3].max():
                                    self.indexes.append((idx, b1, b2, b3))
                    idx += 1




    def __len__(self):
        return len(self.indexes) // self.batch_size

    def get_len(self):
        return len(self.indexes)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        shuffle(self.indexes)


    def __shape__(self):
        data = self.__getitem__(0)
        return data.shape

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        with threading.Lock():
            # Generate indices of the batch
            indexes = [self.indexes[i] for i in range(idx * self.batch_size, (idx + 1) * self.batch_size)]
            im_fix = np.stack([self.preprocess(self.pairs[idx[0]][0][idx[1]][...,idx[2]]) for idx in indexes])[...,None]
            im_mov = np.stack([self.preprocess(self.pairs[idx[0]][1][idx[1]][...,idx[3]]) for idx in indexes])[...,None]
            im_fix = tf.identity(im_fix)
            im_mov = tf.identity(im_mov)
            lbl_fix = np.stack([self.pairs[idx[0]][2][idx[1]] for idx in indexes])[...,None]
            lbl_mov = np.stack([self.pairs[idx[0]][3][idx[1]] for idx in indexes])[...,None]
            lbl_fix = tf.identity(lbl_fix)
            lbl_mov = tf.identity(lbl_mov)

            if self.ft_training:
                param_fix = np.stack([self.preprocess(self.pairs[idx[0]][4][idx[1]][...,idx[2]]) for idx in indexes])[..., None]
                param_mov = np.stack([self.preprocess(self.pairs[idx[0]][5][idx[1]][...,idx[3]]) for idx in indexes])[..., None]
                param_fix = tf.identity(param_fix)
                param_mov = tf.identity(param_mov)
                return (im_fix, im_mov, lbl_mov, param_mov), (im_fix, lbl_fix, param_fix, im_fix, im_fix)

            else:
                if self.affine:
                    return (im_fix, im_mov, lbl_mov), (im_fix, lbl_fix)
                else:
                    if self.weighted:
                        if self.type_mask == 'bbox':
                            weight_map_fix = np.stack([bounding_box_mask(self.pairs[idx[0]][2][idx[1]]) for idx in indexes])
                            weight_map_mov = np.stack([bounding_box_mask(self.pairs[idx[0]][3][idx[1]]) for idx in indexes])
                            weight_map = np.logical_or(weight_map_fix, weight_map_mov).astype(np.float32)[..., None]
                            weight_map = tf.identity(weight_map)
                        elif self.type_mask == 'gaussian':
                            weight_map = self.generate_weight_map(lbl_fix+lbl_mov).astype(np.float32)
                            weight_map = tf.identity(weight_map)
                        else:
                            raise ImportError("type mask must be bbox or gaussian")

                        return (im_fix, im_mov, lbl_mov, weight_map), (weight_map * im_fix, im_fix, lbl_fix, im_fix, im_fix)
                    else:

                        return (im_fix, im_mov, lbl_mov), (im_fix, lbl_fix, im_fix, im_fix)


    def generate_weight_map(self, mask):
        decay_rate = 0.1
        seg_map = np.copy(mask)
        seg_map[seg_map > 0] = 1
        seg_map_inv = ~seg_map.astype(bool)
        distance_map = edt(seg_map_inv)
        weight_map = np.exp(-decay_rate * distance_map)
        return weight_map.astype(np.float32)

    def preprocess(self, im):
        return normalize(im)




def transpose_array(arr):
    if arr.ndim == 3:
        return np.transpose(arr, (2,0,1))
    elif arr.ndim == 4:
        return np.transpose(arr, (2, 0,1, 3))
    else:
        raise ValueError("Input array must be 3D or 4D.")


def normalize_img_zmean(img):
    ''' Zero mean unit standard deviation normalization based on a mask'''
    mean_ = img.mean()
    std_ = img.std()
    img = (img - mean_) / std_
    return img


def normalize_img(img):
    img = (img - img.min()) / (img.max() - img.min())
    return img

def normalize(img):
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:  # Avoid division by zero
        return (img - img_min) / (img_max - img_min)
    else:
        return np.zeros_like(img)



def bounding_box_mask(mask, margin=5):
    """
    Generate a binary mask of the bounding box around a segmentation mask.

    Parameters:
        mask (np.ndarray): A 2D binary segmentation mask (values: 0 or 1).
        margin (int): Margin to add around the bounding box.

    Returns:
        np.ndarray: A binary mask of the same shape as the input with the bounding box.
    """
    # Find the indices of non-zero (True) elements in the mask
    rows, cols = np.nonzero(mask)

    # If the mask is empty, return an array of zeros
    if len(rows) == 0 or len(cols) == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    # Calculate the bounding box
    x_min, x_max = cols.min(), cols.max()
    y_min, y_max = rows.min(), rows.max()

    # Add the margin
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(mask.shape[1] - 1, x_max + margin)
    y_max = min(mask.shape[0] - 1, y_max + margin)

    # Create an array to represent the bounding box
    bbox_mask = np.zeros_like(mask, dtype=np.uint8)
    bbox_mask[y_min:y_max + 1, x_min:x_max + 1] = 1

    return bbox_mask


def guassian_mask(mask, margin=5):
    """
    Generate a binary mask of the bounding box around a segmentation mask.

    Parameters:
        mask (np.ndarray): A 2D binary segmentation mask (values: 0 or 1).
        margin (int): Margin to add around the bounding box.

    Returns:
        np.ndarray: A binary mask of the same shape as the input with the bounding box.
    """
    # Find the indices of non-zero (True) elements in the mask
    rows, cols = np.nonzero(mask)

    # If the mask is empty, return an array of zeros
    if len(rows) == 0 or len(cols) == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    # Calculate the bounding box
    x_min, x_max = cols.min(), cols.max()
    y_min, y_max = rows.min(), rows.max()

    # Add the margin
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(mask.shape[1] - 1, x_max + margin)
    y_max = min(mask.shape[0] - 1, y_max + margin)

    # Create an array to represent the bounding box
    bbox_mask = np.zeros_like(mask, dtype=np.uint8)
    bbox_mask[y_min:y_max + 1, x_min:x_max + 1] = 1

    return bbox_mask