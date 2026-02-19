#!/usr/bin/env python
"""
Segmentation Fine-tuning Script

This script fine-tunes a pre-trained encoder for kidney segmentation.
It supports two tasks:
    - 'volume': 3-class segmentation (background, right kidney, left kidney)
    - 'cortex': 5-class segmentation (background, right cortex, left cortex, 
                right medulla, left medulla)

Usage:
    python finetune.py [--debug] [--gpu GPU_ID]

Example:
    python finetune.py --debug --gpu 3
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import wandb
from tensorflow.keras import optimizers

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

import configs.finetune as cfg
from data.train_seg_Data_Generator import DataLoader
from losses.ft_losses import lossObj
from losses.reg_losses import dice_loss
from models.model_seg_gn import modelObj
from utils import setup_TF_environment, get_callbacks_save_best, get_logger

# Setup logging
logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Segmentation Fine-tuning')

    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode (limited data, no callbacks)')

    parser.add_argument('--gpu', type=str, default='3',
                        help='GPU ID to use (e.g., "0" or "0,1")')

    parser.add_argument('--fold', type=int, default=None,
                        help='Override fold number from config')

    parser.add_argument('--task', type=str, default=None,
                        choices=['volume', 'cortex'],
                        help='Override task from config')

    return parser.parse_args()


def setup_wandb(config, debug):
    """Initialize Weights & Biases for experiment tracking."""
    if debug:
        logger.info("Debug mode: Skipping WandB initialization")
        return

    try:
        os.environ["WANDB_API_KEY"] = config.wandb_key
        wandb.init(
            project="Renal_fMRI",
            entity=config.wandb_entity,
            group="segmentation",
            name=f'{config.experiment_name}_fold{config.fold}',
            config={
                'batch_size': config.batch_size,
                'lr': config.lr_pretrain,
                'num_classes': config.num_classes,
                'task': config.task,
                'contrast': config.contrast,
                'fold': config.fold
            },
            settings=wandb.Settings(start_method='thread')
        )
        logger.info(f"WandB initialized: {config.experiment_name}_fold{config.fold}")
    except Exception as e:
        logger.error(f"Failed to initialize WandB: {e}")
        logger.warning("Continuing without WandB tracking...")


def create_model(config, checkpoint_path=None):
    """
    Create segmentation model and load pre-trained weights if provided.

    Args:
        config: Configuration object
        checkpoint_path: Path to pre-trained encoder weights

    Returns:
        Compiled Keras model
    """
    logger.info(f"Creating U-Net for {config.num_classes}-class segmentation")

    # Initialize model utility
    mm_utils = modelObj(config)

    # Create segmentation U-Net
    model = mm_utils.seg_unet(num_classes=config.num_classes)

    # Load pre-trained weights if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading pre-trained weights from: {checkpoint_path}")
        model.load_weights(checkpoint_path, by_name=True)
    elif checkpoint_path:
        logger.warning(f"Checkpoint not found: {checkpoint_path}")

    # Print model summary in debug mode
    if hasattr(config, 'debug') and config.debug:
        model.summary()

    return model


def setup_loss_and_optimizer(config):
    """
    Setup loss function and optimizer.

    Returns:
        Tuple of (loss_function, optimizer)
    """
    logger.info("Setting up loss and optimizer")

    # Initialize loss object
    loss = lossObj()

    # Select appropriate loss based on number of classes
    if config.num_classes == 3:
        custom_loss = loss.tversky_loss
        logger.info("Using Tversky loss for 3-class segmentation")
    else:
        custom_loss = loss.tversky_loss_lbl5
        logger.info("Using Tversky loss for 5-class segmentation")

    # Create optimizer with gradient clipping
    optimizer = optimizers.Adam(
        learning_rate=config.lr_pretrain,
        clipvalue=1.0  # Gradient clipping for stability
    )

    return custom_loss, optimizer


def create_data_loaders(config, debug):
    """
    Create training and validation data loaders.

    Args:
        config: Configuration object
        debug: Whether in debug mode

    Returns:
        Tuple of (train_generator, val_generator)
    """
    logger.info("Creating data loaders...")

    # Training data loader
    train_gen = DataLoader(config, debug=debug, train_flag=True)
    logger.info(f"Training samples: {len(train_gen)} batches, {train_gen.get_len()} total samples")

    # Validation data loader
    val_gen = DataLoader(config, debug=debug, train_flag=False)
    logger.info(f"Validation samples: {len(val_gen)} batches, {val_gen.get_len()} total samples")

    return train_gen, val_gen


def setup_output_directories(config):
    """
    Create output directories for saving models and logs.

    Returns:
        Tuple of (save_dir, csv_path)
    """
    # Create experiment-specific save directory
    save_dir = os.path.join(config.base_save_dir, config.experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    # Path for training log CSV
    csv_path = os.path.join(save_dir, 'training.log')

    logger.info(f"Output directory: {save_dir}")
    logger.info(f"Log file: {csv_path}")

    return save_dir, csv_path


def save_config(config, save_dir):
    """Save configuration to a text file."""
    config_path = os.path.join(save_dir, 'finetune_config_params.txt')

    with open(config_path, 'w') as f:
        f.write(f"# Segmentation Fine-tuning Configuration\n")
        f.write(f"# Generated: {__import__('datetime').datetime.now()}\n")
        f.write("#" + "=" * 60 + "\n\n")

        for name, value in sorted(config.__dict__.items()):
            if not name.startswith('_') and not callable(value):
                f.write(f"{name} = {repr(value)}\n")

    logger.info(f"Configuration saved to: {config_path}")


def main(debug=False, gpu_id="3", fold_override=None, task_override=None):
    """
    Main training function.

    Args:
        debug: Run in debug mode
        gpu_id: GPU ID to use
        fold_override: Override fold number
        task_override: Override task type
    """
    # Setup GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    logger.info(f"Using GPU(s): {gpu_id}")

    if task_override is not None:
        cfg.task = task_override
        cfg.num_classes = 3 if task_override == 'volume' else 5
        cfg.experiment_name = f'seg_{cfg.tgt_contrast}_nc_{cfg.num_classes}_no_pretraining'
        logger.info(f"Overriding task: {task_override}")

    # Store debug flag in config
    cfg.debug = debug

    # Setup TensorFlow environment
    setup_TF_environment(cfg.gpus_available)

    # Initialize WandB
    setup_wandb(cfg, debug)

    try:
        # Step 1: Create data loaders
        logger.info("=" * 60)
        logger.info("STEP 1: Creating data loaders")
        logger.info("=" * 60)
        train_gen, val_gen = create_data_loaders(cfg, debug)

        # Step 2: Create model
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Creating model")
        logger.info("=" * 60)
        model = create_model(cfg, cfg.checkpoint_path)

        # Step 3: Setup loss and optimizer
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Setting up loss and optimizer")
        logger.info("=" * 60)
        loss_fn, optimizer = setup_loss_and_optimizer(cfg)

        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[dice_loss().loss]  # Add Dice coefficient as metric
        )

        # Step 4: Setup output directories
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Setting up output directories")
        logger.info("=" * 60)
        save_dir, csv_path = setup_output_directories(cfg)

        # Step 5: Setup callbacks
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: Setting up callbacks")
        logger.info("=" * 60)

        if debug:
            callbacks = []
            logger.info("Debug mode: No callbacks")
        else:
            callbacks = get_callbacks_save_best(
                csv_path=csv_path,
                checkpoint_dir=save_dir,
                monitor='val_loss',
                mode='min'
            )
            logger.info(f"Callbacks: {[c.__class__.__name__ for c in callbacks]}")

        # Step 6: Train model
        logger.info("\n" + "=" * 60)
        logger.info("STEP 6: Starting training")
        logger.info("=" * 60)
        logger.info(f"Epochs: {cfg.initial_epoch} to {cfg.num_epochs}")
        logger.info(f"Batch size: {cfg.batch_size}")
        logger.info(f"Task: {cfg.task} ({cfg.num_classes} classes)")

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=cfg.num_epochs,
            verbose=1,
            workers=6,
            callbacks=callbacks,
            initial_epoch=cfg.initial_epoch,
            use_multiprocessing=False,
            shuffle=True
        )

        # Step 7: Save final model
        logger.info("\n" + "=" * 60)
        logger.info("STEP 7: Saving final model")
        logger.info("=" * 60)

        final_weights_path = os.path.join(save_dir, f'weights_{cfg.num_epochs}.hdf5')
        model.save_weights(
            final_weights_path,
            overwrite=True,
            save_format='h5'
        )
        logger.info(f"Final model saved to: {final_weights_path}")

        # Step 8: Save configuration
        save_config(cfg, save_dir)

        # Training summary
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Final epoch: {cfg.num_epochs}")
        logger.info(f"Best validation loss: {min(history.history.get('val_loss', [0])):.4f}")

    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise

    finally:
        # Cleanup
        if not debug and 'wandb' in sys.modules:
            wandb.finish()
            logger.info("WandB run finished")


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Log startup information
    logger.info("=" * 60)
    logger.info("SEGMENTATION FINE-TUNING")
    logger.info("=" * 60)
    logger.info(f"Debug mode: {args.debug}")
    logger.info(f"GPU ID: {args.gpu}")
    logger.info(f"Fold override: {args.fold}")
    logger.info(f"Task override: {args.task}")

    # Run main training
    main(
        debug=args.debug,
        gpu_id=args.gpu,
        fold_override=args.fold,
        task_override=args.task
    )