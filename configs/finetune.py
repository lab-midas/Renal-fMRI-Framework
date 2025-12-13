gpus_available = '3'

''' Data params'''
img_size_x = 256
img_size_y = 256
dataset = 'renfi'
num_channels = 1  


wandb_key = 'wandbkey'
wandb_entity = "wandbentity"

''' Training params'''
batch_size = 12
lr_pretrain = 1e-3
latent_dim = 64
initial_epoch = 0
num_epochs = 250

# settings for full/partial decoder
partial_decoder = 0
warm_start = 0


data_dir = 'data_dir'
checkpoint_path = "pretraining_checkpoint_path"

""" Experiment param"""
fold = 5
val_sub = f'<DATA_ROOT>/folds/test_f{fold}.npy'
train_sub = f'<DATA_ROOT>/folds/train_f{fold}.npy'

base_save_dir = f'<LOG_ROOT>/segmentation/folds/fold{fold}'
pretrain_method = 'pretrain_method'
experiment_name = 'experiment_name'




