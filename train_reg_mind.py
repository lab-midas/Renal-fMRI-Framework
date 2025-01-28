import tensorflow as tf
import os
import wandb
from tensorflow.keras import optimizers
import configs.registration_momind as cfg
from utils import setup_TF_environment, get_callbacks
from data.train_reg_Data_Generator import DataLoaderPairwise
from models.model_reg_gn import modelObj
from losses.reg_losses import  design_loss, smooth, residuce_loss, dice_loss, MAD, Grad


def main(debug):
    setup_TF_environment(cfg.gpus_available)
    if not debug:
        os.environ["WANDB_API_KEY"] = cfg.wandb_key
        wandb.init(project="Renal_fMRI",
                   entity=cfg.wandb_entity,
                   group="registration",
                   name=cfg.experiment_name,
                   settings=wandb.Settings(start_method='thread'))
    ''' Load train data'''
    train_gen = DataLoaderPairwise(cfg, debug, train_flag=True)
    x = train_gen.__getitem__(1)
    val_gen = DataLoaderPairwise(cfg, debug, train_flag=False)

    # non-rigid
    customLoss = [design_loss(parameter_mi=cfg.para_mi, parameter=cfg.para_jl).mi_clipmse, dice_loss().loss, MAD().loss, Grad().loss, residuce_loss]
    loss_weights = [0, 0, 10, 1, 0]

    AdamOpt = optimizers.Adam(learning_rate=cfg.lr_pretrain, clipvalue=1.0)

    mm_utils = modelObj(cfg)
    ae_reg = mm_utils.reg_unet_mind(cfg.weighted)
    print(ae_reg.summary())

    if cfg.checkpoint_path:
        ae_reg.load_weights(cfg.checkpoint_path, by_name=True)
    ae_reg.compile(optimizer=AdamOpt, loss=customLoss, loss_weights=loss_weights, metrics={'stn_mov_lbl': dice_loss().loss})

    # %% Generate save_dir
    wts_save_dir = os.path.join(cfg.base_save_dir, 'pairwise', f'{cfg.experiment_name}')
    if debug:
        print('logs will be saved in ', wts_save_dir)
    else:
        os.makedirs(wts_save_dir, exist_ok=True)
        print('Creating ', wts_save_dir)
        csvPath = wts_save_dir + 'training.log'

    # %% Create callbacks
    callbacks = [] if debug else get_callbacks(csvPath, wts_save_dir)

    # %% Train the model
    ae_reg.fit(train_gen,
                      validation_data=val_gen,
                      epochs=cfg.num_epochs,
                      verbose=1,
                      workers=6,
                      callbacks=callbacks,
                      initial_epoch=cfg.initial_epoch,
                      use_multiprocessing=False)

    # Save final weights
    ae_reg.save_weights(
        wts_save_dir + '_weights_' + str(cfg.num_epochs) + '.hdf5',
        overwrite=True,
        save_format='h5',
        options=None)
    #
    ## %% Save the configuration for the run
    cfg_txt_name = wts_save_dir + '_finetune_config_params.txt'
    with open(cfg_txt_name, 'w') as f:
        for name, value in cfg.__dict__.items():
            f.write('{} = {!r}\n'.format(name, value))

    wandb.finish()


if __name__ == '__main__':
    debug = False
    if debug:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    main(debug)

