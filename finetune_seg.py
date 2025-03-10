import os
import wandb
from tensorflow.keras import optimizers
import configs.finetune as cfg
from utils import setup_TF_environment, get_callbacks_save_best
from losses.ft_losses import lossObj
from data.train_seg_Data_Generator import DataLoader
from models.model_seg_gn import modelObj
from losses.reg_losses import dice_loss
import os

import wandb
from tensorflow.keras import optimizers

import configs.finetune as cfg
from data.train_seg_Data_Generator import DataLoader
from losses.ft_losses import lossObj
from losses.reg_losses import dice_loss
from models.model_seg_gn import modelObj
from utils import setup_TF_environment, get_callbacks_save_best


def main(debug):
    setup_TF_environment(cfg.gpus_available)
    if not debug:
        os.environ["WANDB_API_KEY"] = cfg.wandb_key
        wandb.init(project="Renal_fMRI",
                   entity=cfg.wandb_entity,
                   group="pretraining",
                   name=f'{cfg.experiment_name}_fold{cfg.fold}',
                   settings=wandb.Settings(start_method='thread'))
    ''' Load train data'''
    train_gen = DataLoader(cfg, debug=debug, train_flag=True)
    val_gen = DataLoader(cfg, debug=debug, train_flag=False)


    loss = lossObj()
    customLoss = loss.tversky_loss if cfg.num_classes == 3 else loss.tversky_loss_lbl5
    print(customLoss)
    AdamOpt = optimizers.Adam(learning_rate=cfg.lr_pretrain, clipvalue=1.0)

    mm_utils = modelObj(cfg)
    ae_pretrained = mm_utils.seg_unet(num_classes=cfg.num_classes)
    if cfg.checkpoint_path:
        ae_pretrained.load_weights(cfg.checkpoint_path, by_name=True)
    ae_pretrained.compile(optimizer=AdamOpt, loss=customLoss, metrics=dice_loss().loss)

    # %% Generate save_dir
    wts_save_dir = os.path.join(cfg.base_save_dir, f'{cfg.experiment_name}')
    os.makedirs(wts_save_dir, exist_ok=True)
    print('Creating ', wts_save_dir)
    csvPath = wts_save_dir + 'training.log'

    # %% Create callbacks
    callbacks =  [] if debug else get_callbacks_save_best(csvPath, wts_save_dir) #

    # %% Train the model
    ae_pretrained.fit(train_gen,
                      validation_data=val_gen,
                      epochs=cfg.num_epochs,
                      verbose=1,
                      workers=6,
                      callbacks=callbacks,
                      initial_epoch=cfg.initial_epoch,
                      use_multiprocessing=False)

    # Save final weights
    ae_pretrained.save_weights(
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

    #wandb.finish()

if __name__ == '__main__':
    debug = False
    if debug:
        os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    main(debug)

