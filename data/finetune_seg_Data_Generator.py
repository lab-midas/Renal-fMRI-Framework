import itertools
import os
import random
import threading
from random import shuffle

import albumentations as A
import numpy as np
import tensorflow as tf
import tensorflow.keras


class DataLoaderObj(tensorflow.keras.utils.Sequence):
    def __init__(self, cfg, train_flag=True):
        self.patch_size = cfg.patch_size
        self.batch_size = cfg.batch_size
        self.num_channels = cfg.num_channels
        self.cfg = cfg
        self.contrast_list = cfg.contrast_list
        self.num_labels = cfg.num_classes
        self.num_constraints = 2 if cfg.use_mask_sampling else 1
        self.num_clusters = cfg.num_samples_loss_eval
        self.train_flag = train_flag
        self.size = (cfg.img_size_x, cfg.img_size_y)
        if train_flag:
            print('Initializing training dataloader')
            self.input_hdf5_img     = cfg.train_img_dir
            self.input_hdf5_lbl = cfg.train_lbl_dir

            self.aug = A.OneOf([A.NoOp(p=0.1),
                                A.Compose([A.ElasticTransform(border_mode=0, alpha=20, sigma=20, alpha_affine=20 * 0.03,
                                                              p=0.3),
                                           A.RandomBrightnessContrast(p=0.3),
                                           A.OneOf([
                                               A.Affine(translate_percent=(0, 0.15), p=0.5),
                                               A.Affine(scale=(0.9, 1), p=0.5), ], p=0.5),
                                           A.RandomResizedCrop(height=self.size[0], width=self.size[1], scale=(0.8, 1),
                                                               p=0.3)], p=0.9)], p=1)
        else:
            print('Initializing validation dataloader')
            self.input_hdf5_img     = cfg.val_img_dir
            self.input_hdf5_lbl = cfg.val_lbl_dir

        self.img, self.param_cluster = [], []
        self.files_imgs = os.listdir(self.input_hdf5_img)
        if self.contrast_list:
            self.files_imgs = [file for contrast in self.contrast_list for file in self.files_imgs if (contrast in file)]
        print(self.files_imgs)
        files_param = [a.replace('img', 'lbl') for a in self.files_imgs]
        print('### started loading images')
        img_paths = [os.path.join(self.input_hdf5_img, img_name) for img_name in self.files_imgs]
        self.img = [np.load(img_path, mmap_mode='r') for img_path in img_paths]
        print(self.img[0].shape)
        print('### started loading labels')
        params_paths = [os.path.join(self.input_hdf5_lbl, param_name) for param_name in files_param]
        self.labels = [np.load(param_path, mmap_mode='r') for param_path in params_paths]
        print('### params and images loaded')
        assert (len(self.img) == len(self.labels))


        self.len_of_data = sum([img.shape[0]*img.shape[-1] for img in self.img])
        self.num_samples = self.len_of_data // 1  # use it to control size of sampels per epoch
        print('Total samples :', self.num_samples)
        self.img_size_x = self.img[0].shape[1]
        self.img_size_y = self.img[0].shape[2]
        self.arr_indexes = self.get_indexes()

    def get_indexes(self):
        shapes = [(i, img.shape[0], img.shape[-1]) for i, img in enumerate(self.img)]
        combinations = []
        if self.num_channels == 1:
            for ind, slices, channels in shapes:
                for s, c in itertools.product(range(slices), range(channels)):
                    combinations.append((ind, s, c))
        else:
            for ind, slices, channels in shapes:
                for s in range(slices):
                    combinations.append((ind, s, range(channels)))
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
            indexes = self.arr_indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
            x_train = self.generate_X(indexes)
            y_train = self.generate_mask(indexes)
            if self.train_flag:
                random.seed(7)
                augmented = [self.aug(image=x_train[i], mask = y_train[i]) for i in range(self.batch_size)]
                x_train = np.stack([augmented[i]['image'] for i in range(self.batch_size)], axis=0)
                y_train = np.stack([augmented[i]['mask'] for i in range(self.batch_size)], axis=0)
            #print(y_train.shape)
            x_train = tf.identity(x_train)
            y_train = tf.identity(y_train)
            return x_train, y_train

    def generate_X(self, list_idx):
        X = np.zeros((self.batch_size, self.img_size_x, self.img_size_y, self.num_channels),dtype=np.float32)
        for jj in range(0, self.batch_size):
            idx, b, c = list_idx[jj]
            X[jj] = self.img[idx][b,:,:,c][..., None]
        return X

    def generate_mask(self, list_idx):
        X = np.zeros((self.batch_size, self.img_size_x, self.img_size_y, self.num_labels),dtype="float32")
        for jj in range(0, self.batch_size):
            idx, b, c = list_idx[jj]
            X[jj] = self.labels[idx][b]
        return X.astype(np.float32)



