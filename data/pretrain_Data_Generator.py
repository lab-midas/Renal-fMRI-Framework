import itertools
import os
import threading
from random import shuffle

import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow.keras
from skimage.util import view_as_blocks


class DataLoaderObj(tensorflow.keras.utils.Sequence):
    ''' Config cfg contains the parameters that control training'''
    def __init__(self, cfg, debug, train_flag=True):
        self.patch_size = cfg.patch_size
        self.batch_size = cfg.batch_size
        self.num_channels = cfg.num_channels
        self.cfg = cfg
        self.ft_param = cfg.ft_param
        self.contrast = cfg.contrast
        self.num_constraints = 2 if cfg.use_mask_sampling else 1
        self.num_clusters = cfg.num_samples_loss_eval
        self.train_flag = train_flag
        if train_flag:
            print('Initializing training dataloader')
            self.input_img     = np.load(cfg.train_sub).tolist()
        else:
            print('Initializing validation dataloader')
            self.input_img     = np.load(cfg.val_sub).tolist()
        if debug:
            self.input_img = self.input_img[:3]
        self.data_dir = cfg.data_dir
        self.img, self.param_cluster = [], []
        if self.ft_param:
            self.ft = []

        print('### number of files', len(self.input_img))
        print('### started loading images')
        self.load_ft() if self.ft_param else self.load_img_cluster()
        print('### params and images loaded')
        self.len_of_data = sum([img.shape[0]*img.shape[-1] for img in self.img])
        self.num_samples = self.len_of_data // 1  # use it to control size of sampels per epoch
        print('Total samples :', self.num_samples)
        self.img_size_x = cfg.img_size_x
        self.img_size_y = cfg.img_size_y
        self.arr_indexes = self.get_indexes()

    def load_ft(self):
        for sub in self.input_img:
            for parameter in self.contrast:
                path = os.path.join(self.data_dir, parameter, f'Constraint_map_{sub}_20.mat')
                if os.path.exists(path):
                    f = sio.loadmat(path)
                    img = f['img'].astype('float32')
                    ft = f['param'].astype('float32')
                    mask = f['mask'].astype('float32')
                    mask[mask == 0] = 5

                    img = np.transpose(img, (2, 0, 1, 3)) if img.ndim == 4 else np.transpose(img, (2, 0, 1))[..., None]
                    ft = np.transpose(ft, (2, 0, 1, 3))
                    mask = np.tile(np.transpose(mask, (2, 0, 1))[..., None],(1, 1, 1, img.shape[-1]))

                    self.img.append(img)
                    self.param_cluster.append(mask)
                    self.ft.append(ft)

    def load_img_cluster(self):
        for sub in self.input_img:
            for parameter in self.contrast:
                path = os.path.join(self.data_dir, parameter, f'Constraint_map_{sub}_20.mat')
                if os.path.exists(path):
                    f = sio.loadmat(path)
                    img = f['img'].astype('float32')
                    map = f['param'].astype('float32')
                    img = np.transpose(img, (2, 0, 1, 3)) if img.ndim == 4 else np.transpose(img, (2, 0, 1))[...,None]
                    map = np.transpose(map, (2, 0, 1))
                    self.img.append(img)
                    self.param_cluster.append(map)
                else:
                    print(path, ' not found')

    def get_indexes(self):
        shapes = [(i, img.shape[0], img.shape[-1]) for i, img in enumerate(self.img)]
        combinations = []
        for ind, slices, channels in shapes:
            for s, c in itertools.product(range(slices), range(channels)):
                combinations.append((ind, s, c))
        shuffle(combinations)
        return combinations


    def __len__(self):
        return len(self.arr_indexes) // self.batch_size

    def get_len(self):
        return len(self.arr_indexes)
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        shuffle(self.arr_indexes)
        
        
    def __shape__(self):
        data = self.__getitem__(0)
        return data.shape
    
    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        with threading.Lock():
            # Generate indices of the batch
            idx = self.arr_indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
            x_train = self.generate_X(idx)
            x_train = tf.identity(x_train)
            if self.ft_param:
                y_param_cluster = self.generate_clusters(idx, self.param_cluster)
                y_mind = self.generate_clusters(idx, self.ft)
                y_train = (tf.identity(y_param_cluster), tf.identity(y_mind))
            else:
                y_train = self.generate_clusters(idx, self.param_cluster)
                y_train = tf.identity(y_train)
            return x_train, y_train
        
    def generate_X(self, list_idx):
        X = np.zeros((self.batch_size, self.img_size_x, self.img_size_y, self.num_channels),dtype="float64")
        for jj in range(0, self.batch_size):
            idx, b, c = list_idx[jj]
            X[jj] = self.img[idx][b,...,c][...,None]
        return X
    def generate_mask(self, X):
        ''' Generates a mask from foreground regions along 1st channel of image'''
        mask = np.zeros(X.shape)
        mask[ X > X.min()] = 1
        return mask

    def generate_clusters(self, list_idx, param):
        patch_size = self.patch_size
        batch_size = self.batch_size
        num_constraints = self.num_constraints

        X = np.zeros((self.batch_size, self.img_size_x, self.img_size_y, self.num_channels), dtype="float64")
        Y = np.zeros((self.img_size_x, self.img_size_y, num_constraints, batch_size), dtype="int64")
        mask = np.zeros((self.img_size_x, self.img_size_y, batch_size), dtype="int64")
        ''' Note that batch dim is transposed in Y for view as blocks function '''

        for jj in range(0, self.batch_size):
            idx, b, c = list_idx[jj]
            temp = self.img[idx][b,...,c][...,None]
            X[jj] = temp
            mask[..., jj] = self.generate_mask(temp[..., 0])
            param_in = param[idx][b] if param[idx].ndim == 3 else param[idx][b,...,c]
            Y[..., 0, jj] = np.squeeze(param_in)
            Y[..., -1, jj] = mask[..., jj]

        # This section identifies the majority class in non-overlapping patch_size x patch_size regions of the constraint map
        if patch_size > 1:
            y_train_blk = view_as_blocks(Y, (patch_size, patch_size, num_constraints, batch_size)).squeeze()
            xDim, yDim = y_train_blk.shape[0], y_train_blk.shape[1]
            y_train_blk = np.reshape(y_train_blk, (xDim, yDim, patch_size * patch_size, num_constraints, batch_size))
            y_train = np.apply_along_axis(self.get_freq_labels, -3, y_train_blk)
        else:
            y_train = Y

        for jj in range(0, self.batch_size):
            temp = y_train[..., -1, jj].squeeze()  # get the mask
            y_train[..., -1, jj] = self.generate_indices_from_mask(temp)  # generate indices from mask

        ''' Note that batch dim was transposed in Y for view as blocks function '''
        #print(y_train.shape)
        Y = np.transpose(y_train, (3, 0, 1, 2))
        return Y

    def get_freq_labels(self, arr):
        return np.bincount(arr).argmax()

    def generate_indices_from_mask(self, mask):
        ''' Returns a mask with random indices from the brain/image for patchwise CCL loss calculation
        '''
        xDim, yDim = mask.shape
        mask_f = np.ndarray.flatten(mask.copy())
        all_idx = np.where(mask_f == 1)[0]
        np.random.shuffle(all_idx)
        temp_idx = all_idx[:self.num_clusters]
        idx_mask = np.zeros(mask_f.shape, dtype='int64')
        idx_mask[temp_idx] = 1
        mask = np.reshape(idx_mask, (xDim, yDim))
        return mask


class DataLoaderObj3D(tensorflow.keras.utils.Sequence):
    ''' Config cfg contains the parameters that control training'''

    def __init__(self, cfg, debug, train_flag=True):
        self.patch_size = cfg.patch_size
        self.batch_size = cfg.batch_size
        self.num_channels = cfg.num_channels
        self.cfg = cfg
        self.ft_param = False
        self.contrast = cfg.contrast
        self.num_constraints = 2 if cfg.use_mask_sampling else 1
        self.num_clusters = cfg.num_samples_loss_eval
        self.train_flag = train_flag
        if train_flag:
            print('Initializing training dataloader')
            self.input_img = np.load(cfg.train_sub).tolist()
        else:
            print('Initializing validation dataloader')
            self.input_img = np.load(cfg.val_sub).tolist()
        if debug:
            self.input_img = self.input_img[:3]
        self.data_dir = cfg.data_dir
        self.img, self.param_cluster = [], []

        print('### number of files', len(self.input_img))
        print('### started loading images')
        self.load_ft() if self.ft_param else self.load_img_cluster()
        print('### params and images loaded')
        self.len_of_data = sum([img.shape[-1] for img in self.img])
        self.num_samples = self.len_of_data // 1  # use it to control size of sampels per epoch
        print('Total samples :', self.num_samples)
        self.img_size_x = cfg.img_size_x
        self.img_size_y = cfg.img_size_y
        self.img_size_z = cfg.img_size_z
        self.arr_indexes = self.get_indexes()

    def load_ft(self):
        for sub in self.input_img:
            for parameter in self.contrast:
                path = os.path.join(self.data_dir, parameter, f'Constraint_map_{sub}_20.mat')
                if os.path.exists(path):
                    f = sio.loadmat(path)
                    img = f['img'].astype('float32')
                    ft = f['param'].astype('float32')
                    mask = f['mask'].astype('float32')
                    mask[mask == 0] = 5

                    img = np.transpose(img, (2, 0, 1, 3)) if img.ndim == 4 else np.transpose(img, (2, 0, 1))[..., None]
                    ft = np.transpose(ft, (2, 0, 1, 3))
                    mask = np.tile(np.transpose(mask, (2, 0, 1))[..., None], (1, 1, 1, img.shape[-1]))

                    self.img.append(img)
                    self.param_cluster.append(mask)
                    self.ft.append(ft)

    def load_img_cluster(self):
        for sub in self.input_img:
            for parameter in self.contrast:
                path = os.path.join(self.data_dir, parameter, f'Constraint_map_{sub}_20.mat')
                if os.path.exists(path):
                    f = sio.loadmat(path)
                    img = f['img'].astype('float32')
                    map = f['param'].astype('float32')
                    img = np.transpose(img, (2, 0, 1, 3)) if img.ndim == 4 else np.transpose(img, (2, 0, 1))[..., None]
                    map = np.transpose(map, (2, 0, 1))
                    self.img.append(img)
                    self.param_cluster.append(map)
                else:
                    print(path, ' not found')

    def get_indexes(self):
        shapes = [(i, img.shape[-1]) for i, img in enumerate(self.img)]
        # print(shapes)
        combinations = []
        for ind, channels in shapes:
            for c in range(channels):
                combinations.append((ind, c))
        shuffle(combinations)
        return combinations

    def __len__(self):
        return len(self.arr_indexes) // self.batch_size

    def get_len(self):
        return len(self.arr_indexes)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        shuffle(self.arr_indexes)

    def __shape__(self):
        data = self.__getitem__(0)
        return data.shape

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        with threading.Lock():
            # Generate indices of the batch
            idx = self.arr_indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
            x_train = self.generate_X(idx)
            x_train = tf.identity(x_train)

            y_train = self.generate_clusters(idx, self.param_cluster)
            y_train = tf.identity(y_train)
            return x_train, y_train

    def generate_X(self, list_idx):
        X = np.zeros((self.batch_size, self.img_size_x, self.img_size_y, self.img_size_z, self.num_channels), dtype="float64")
        for jj in range(0, self.batch_size):
            idx, c = list_idx[jj]
            X[jj] = np.transpose(self.img[idx][..., c][..., None],(1,2,0,3))
        return X

    def generate_mask(self, X):
        ''' Generates a mask from foreground regions along 1st channel of image'''
        mask = np.zeros(X.shape)
        mask[X > X.min()] = 1
        return mask

    def generate_clusters(self, list_idx, param):
        patch_size = self.patch_size
        batch_size = self.batch_size
        num_constraints = self.num_constraints

        X = np.zeros((self.batch_size, self.img_size_x, self.img_size_y, self.img_size_z, self.num_channels), dtype="float64")
        Y = np.zeros((self.img_size_x, self.img_size_y, self.img_size_z, num_constraints, batch_size), dtype="int64")
        mask = np.zeros((self.img_size_x, self.img_size_y, self.img_size_z, batch_size), dtype="int64")
        ''' Note that batch dim is transposed in Y for view as blocks function '''

        for jj in range(0, self.batch_size):
            idx, c = list_idx[jj]
            temp = self.img[idx][..., c][..., None]
            temp = np.transpose(temp,(1,2,0,3))
            X[jj] = temp
            mask[..., jj] = self.generate_mask(temp[..., 0])
            param_in = param[idx] if param[idx].ndim == 3 else param[idx][..., c]
            param_in = np.transpose(param_in, (1, 2, 0))
            Y[..., 0, jj] = np.squeeze(param_in)
            Y[..., -1, jj] = mask[..., jj]

        # This section identifies the majority class in non-overlapping patch_size x patch_size regions of the constraint map
        if patch_size > 1:
            #print('Y',Y.shape, patch_size)
            y_train_blk = view_as_blocks(Y, (patch_size, patch_size, patch_size, num_constraints, batch_size)).squeeze()
            #print('y_train_blk', y_train_blk.shape, patch_size)
            xDim, yDim, zDim = y_train_blk.shape[0], y_train_blk.shape[1], y_train_blk.shape[2]
            y_train_blk = np.reshape(y_train_blk, (xDim, yDim, zDim, patch_size * patch_size * patch_size, num_constraints, batch_size))
            y_train = np.apply_along_axis(self.get_freq_labels, -3, y_train_blk)
        else:
            y_train = Y

        for jj in range(0, self.batch_size):
            temp = y_train[..., -1, jj].squeeze()  # get the mask
            y_train[..., -1, jj] = self.generate_indices_from_mask(temp)  # generate indices from mask

        ''' Note that batch dim was transposed in Y for view as blocks function '''
        # print(y_train.shape)
        Y = np.transpose(y_train, (4, 0, 1, 2, 3))
        return Y

    def get_freq_labels(self, arr):
        return np.bincount(arr).argmax()

    def generate_indices_from_mask(self, mask):
        ''' Returns a mask with random indices from the brain/image for patchwise CCL loss calculation
        '''
        xDim, yDim, zDim = mask.shape
        mask_f = np.ndarray.flatten(mask.copy())
        all_idx = np.where(mask_f == 1)[0]
        np.random.shuffle(all_idx)
        temp_idx = all_idx[:self.num_clusters]
        idx_mask = np.zeros(mask_f.shape, dtype='int64')
        idx_mask[temp_idx] = 1
        mask = np.reshape(idx_mask, (xDim, yDim, zDim))
        return mask


