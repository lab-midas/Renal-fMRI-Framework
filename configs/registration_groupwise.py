gpus_available = '3'

''' data params'''
img_size_x = 256
img_size_y = 256

experiment_name = 'experiment_name'
wandb_key = 'wandbkey'
wandb_entity = "wandbentity"

num_channels = 1
zmean = True

''' Training params'''
batch_size = 8
initial_epoch = 0
lr_pretrain = 1e-4
num_epochs = 500
num_classes = 1

''' Loss params'''
ft_training = True

data_dir = '<DATA_ROOT>/groupwise_all'
val_sub = '<DATA_ROOT>/test.npy'
train_sub = '<DATA_ROOT>train_f.npy'


out_labels = True
checkpoint_path = False

""" Experiment param"""
contrast_template = ['DIXON']
contrast_moving = ['DIXON', 'T1', 'T2', 'T2*', 'ASL', 'DWI']
num_contrasts = len(contrast_moving)

out_features = False
clip = False
weighted = True
pca_template = False
affine = False

checkpoint_affine = '<checkpoint_affine>'
base_save_dir = '<base_save_dir>'

