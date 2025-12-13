gpus_available = '3'

''' data params'''
img_size_x = 256
img_size_y = 256
dataset = 'renfi'
contrast = ['T2*', 'T2']
num_channels = 1
zmean = True

''' Training params'''
batch_size = 8
lr_pretrain = 1e-3
latent_dim = 64
initial_epoch = 0
num_epochs = 250

# settings for full/partial decoder
partial_decoder = 0
warm_start = 0

''' Loss params'''
temperature = 0.1
patch_size = 4
topk = 100
num_samples_loss_eval = 20
contrastive_loss_type = 2  # pairwise,   options are 1: setwise, 2: pairwise (recommended)
use_mask_sampling = 1

base_save_dir = f'<CHECKPOINT_ROOT>/pretrain/{name_experiment}'
data_dir = '<DATA_ROOT>'

val_sub = '<DATA_ROOT>/test.npy'
train_sub = '<DATA_ROOT>/train.npy'

checkpoint_path = False

""" Experiment param"""
experiment_name = 'experiment_name'
wandb_key = 'wandbkey'
wandb_entity = "wandbentity"



