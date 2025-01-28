import os
import wandb
from tensorflow.keras import optimizers
from models.model_seg_gn import modelObj
from losses.constrained_contrastive_loss import lossObj
import configs.pretrain_mind as cfg
from utils import setup_TF_environment, get_callbacks
from data.pretrain_Data_Generator import DataLoaderObj



def main(debug):
    setup_TF_environment(cfg.gpus_available)
    if not debug:
        os.environ["WANDB_API_KEY"] = cfg.wandb_key
        wandb.init(project="Renal_fMRI",
                   entity=cfg.wandb_entity,
                   group="pretraining",
                   name=cfg.experiment_name,
                   settings=wandb.Settings(start_method='thread'))

    # %%  Load model
    mm_utils = modelObj(cfg)
    ae_pretrain = mm_utils.encoder_decoder_network(add_PH=True, PH_str='ccl', two_out=True)
    if cfg.checkpoint_path:
        ae_pretrain.load_weights(cfg.checkpoint_path)
        print('loaded checkpoint {}'.format(cfg.checkpoint_path))

    # %% Load the datagenerator.
    train_gen = DataLoaderObj(cfg, debug, train_flag=True)
    if debug:
        a = next(iter(train_gen))
        out = ae_pretrain.predict(a[0])
        print('*** in shape', a[0].shape, '*** out shape', out[0].shape)

    val_gen = DataLoaderObj(cfg, debug, train_flag=False)

    # %% Import loss function and compile model
    loss = lossObj(cfg)
    customLoss = [loss.calc_CCL_batchwise, loss.calc_CCL_batchwise]
    AdamOpt = optimizers.Adam(learning_rate=cfg.lr_pretrain)
    ae_pretrain.compile(optimizer=AdamOpt, loss=customLoss, loss_weights=[0.8, 0.2])

    # %% Generate save_dir
    wts_save_dir = os.path.join(cfg.base_save_dir, cfg.experiment_name)
    os.makedirs(wts_save_dir, exist_ok=True)
    print('Creating ', wts_save_dir)
    csvPath = wts_save_dir + '_training.log'

    # %% Create callbacks
    callbacks = [] if debug else get_callbacks(csvPath, wts_save_dir)

    # %% Train the model
    ae_pretrain.fit(train_gen,
                    epochs=cfg.num_epochs,
                    verbose=1,
                    callbacks=callbacks,
                    workers=6,
                    validation_freq=1,
                    validation_data=val_gen,
                    initial_epoch=cfg.initial_epoch,
                    use_multiprocessing=False)

    # Save final weights
    ae_pretrain.save_weights(
        wts_save_dir + 'weights_' + str(cfg.num_epochs) + '.hdf5',
        overwrite=True,
        save_format='h5',
        options=None)

    # %% Save the configuration for the run
    cfg_txt_name = wts_save_dir + 'config_params.txt'
    with open(cfg_txt_name, 'w') as f:
        for name, value in cfg.__dict__.items():
            f.write('{} = {!r}\n'.format(name, value))

    wandb.finish()

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    debug = False
    main(debug)



