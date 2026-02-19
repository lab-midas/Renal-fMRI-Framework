#!/usr/bin/env python
"""
Constrained Contrastive Learning (CCL) Pre-training Script

This script performs pre-training of a U-Net encoder using constrained contrastive learning.
The pre-trained encoder can later be fine-tuned for segmentation tasks.

Usage:
    python pretrain.py [--debug] [--gpu GPU_ID] [--config CONFIG_PATH]

Example:
    python pretrain.py --debug --gpu 2
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import wandb
import tensorflow as tf
from tensorflow.keras import optimizers

# Add project root to path to ensure imports work
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Local imports
import configs.pretrain as cfg
from data.pretrain_Data_Generator import DataLoaderObj
from losses.constrained_contrastive_loss import lossObj
from models.model_seg_gn import modelObj
from utils import setup_TF_environment, get_callbacks, setup_logging

# Configure logging
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Constrained Contrastive Learning Pre-training')

    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode (limited data, no callbacks)')

    parser.add_argument('--gpu', type=str, default='2',
                        help='GPU ID to use (e.g., "0" or "0,1" for multiple)')

    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom config file (optional)')

    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')

    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Override experiment name from config')

    return parser.parse_args()


def setup_wandb(config: Any, debug: bool) -> None:
    """
    Initialize Weights & Biases for experiment tracking.

    Args:
        config: Configuration object
        debug: Whether in debug mode
    """
    if debug:
        logger.info("Debug mode: Skipping WandB initialization")
        return

    # Check if WandB is configured
    if not hasattr(config, 'wandb_key') or not config.wandb_key:
        logger.warning("WandB API key not found. Skipping experiment tracking.")
        return

    try:
        os.environ["WANDB_API_KEY"] = config.wandb_key
        wandb.init(
            project="Renal_fMRI",
            entity=config.wandb_entity,
            group="pretraining",
            name=config.experiment_name,
            config=config.to_dict() if hasattr(config, 'to_dict') else vars(config),
            settings=wandb.Settings(start_method='thread')
        )
        logger.info(f"WandB initialized for experiment: {config.experiment_name}")
    except Exception as e:
        logger.error(f"Failed to initialize WandB: {e}")
        logger.warning("Continuing without WandB tracking...")


def create_model(config: Any, checkpoint_path: Optional[str] = None) -> tf.keras.Model:
    """
    Create and optionally load pre-trained weights for the model.

    Args:
        config: Configuration object
        checkpoint_path: Path to checkpoint file for resuming training

    Returns:
        Compiled model (without optimizer yet)
    """
    logger.info("Creating U-Net encoder-decoder with projection head")

    # Initialize model utility class
    mm_utils = modelObj(config)

    # Create encoder-decoder with projection head for CCL
    model = mm_utils.encoder_decoder_network(add_PH=True, PH_str='ccl')

    # Load checkpoint if provided
    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            model.load_weights(checkpoint_path)
            logger.info(f"Loaded checkpoint from: {checkpoint_path}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")

    # Print model summary in debug mode
    if hasattr(config, 'debug') and config.debug:
        model.summary()

    return model


def create_data_loaders(config: Any, debug: bool):
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
    train_gen = DataLoaderObj(config, debug, train_flag=True)
    logger.info(f"Training samples: {train_gen.get_len() if hasattr(train_gen, 'get_len') else 'unknown'}")

    # Validation data loader
    val_gen = DataLoaderObj(config, debug, train_flag=False)
    logger.info(f"Validation samples: {val_gen.get_len() if hasattr(val_gen, 'get_len') else 'unknown'}")

    # Debug: test a single batch
    if debug:
        logger.info("Debug mode: Testing data loader...")
        try:
            x_batch, y_batch = next(iter(train_gen))
            logger.info(
                f"Batch shapes - Input: {x_batch.shape}, Output: {y_batch.shape if hasattr(y_batch, 'shape') else 'tuple'}")
        except Exception as e:
            logger.error(f"Error testing data loader: {e}")

    return train_gen, val_gen


def setup_loss_and_optimizer(config: Any):
    """
    Setup loss function and optimizer.

    Args:
        config: Configuration object

    Returns:
        Tuple of (loss_function, optimizer)
    """
    logger.info("Setting up loss function and optimizer...")

    # Initialize loss object
    loss = lossObj(config)
    custom_loss = loss.calc_CCL_batchwise

    # Create optimizer
    optimizer = optimizers.Adam(learning_rate=config.lr_pretrain)

    logger.info(f"Using Adam optimizer with learning rate: {config.lr_pretrain}")
    logger.info(f"Loss type: {'Pairwise' if config.contrastive_loss_type == 2 else 'Setwise'}")

    return custom_loss, optimizer


def setup_output_directories(config: Any) -> tuple:
    """
    Create output directories for saving models and logs.

    Args:
        config: Configuration object

    Returns:
        Tuple of (save_dir, csv_path)
    """
    # Create experiment-specific save directory
    save_dir = os.path.join(config.base_save_dir, config.experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    # Path for training log CSV
    csv_path = os.path.join(save_dir, 'training_log.csv')

    logger.info(f"Output directory: {save_dir}")
    logger.info(f"Log file: {csv_path}")

    return save_dir, csv_path


def save_config(config: Any, save_dir: str) -> None:
    """
    Save configuration to a text file.

    Args:
        config: Configuration object
        save_dir: Directory to save config file
    """
    config_path = os.path.join(save_dir, 'config_params.txt')

    with open(config_path, 'w') as f:
        # Write header
        f.write("# Constrained Contrastive Learning Configuration\n")
        f.write(f"# Generated: {__import__('datetime').datetime.now()}\n")
        f.write("#" + "=" * 60 + "\n\n")

        # Write all config attributes
        for name, value in sorted(config.__dict__.items()):
            # Skip private attributes and modules
            if not name.startswith('_') and not callable(value):
                f.write(f"{name} = {repr(value)}\n")

    logger.info(f"Configuration saved to: {config_path}")


def main(debug: bool = False, gpu_id: str = "2", resume_path: Optional[str] = None):
    """
    Main training function.

    Args:
        debug: Run in debug mode (limited data, no callbacks)
        gpu_id: GPU ID to use
        resume_path: Path to checkpoint to resume from
    """
    # Setup GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    logger.info(f"Using GPU(s): {gpu_id}")

    # Setup TensorFlow environment (memory growth, etc.)
    setup_TF_environment(cfg.gpus_available)

    # Override checkpoint path if resuming
    if resume_path:
        cfg.checkpoint_path = resume_path

    # Store debug flag in config for data loader
    cfg.debug = debug

    # Setup experiment tracking (WandB)
    setup_wandb(cfg, debug)

    try:
        # Step 1: Create model
        logger.info("=" * 60)
        logger.info("STEP 1: Creating model")
        logger.info("=" * 60)
        model = create_model(cfg, cfg.checkpoint_path)

        # Step 2: Create data loaders
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Creating data loaders")
        logger.info("=" * 60)
        train_gen, val_gen = create_data_loaders(cfg, debug)

        # Step 3: Setup loss and optimizer
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Setting up loss and optimizer")
        logger.info("=" * 60)
        loss_fn, optimizer = setup_loss_and_optimizer(cfg)

        # Compile model
        model.compile(optimizer=optimizer, loss=loss_fn)

        # Step 4: Setup output directories
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Setting up output directories")
        logger.info("=" * 60)
        save_dir, csv_path = setup_output_directories(cfg)

        # Step 5: Setup callbacks
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: Setting up callbacks")
        logger.info("=" * 60)
        callbacks = [] if debug else get_callbacks(csv_path, save_dir)

        if not debug:
            logger.info(f"Callbacks configured: {[c.__class__.__name__ for c in callbacks]}")
        else:
            logger.info("Debug mode: No callbacks will be used")

        # Step 6: Train model
        logger.info("\n" + "=" * 60)
        logger.info("STEP 6: Starting training")
        logger.info("=" * 60)
        logger.info(f"Epochs: {cfg.initial_epoch} to {cfg.num_epochs}")
        logger.info(f"Batch size: {cfg.batch_size}")
        logger.info(f"Training samples per epoch: {len(train_gen) * cfg.batch_size}")

        history = model.fit(
            train_gen,
            epochs=cfg.num_epochs,
            verbose=1,
            callbacks=callbacks,
            workers=6,
            validation_freq=1,
            validation_data=val_gen,
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

        # Log training summary
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Final epoch: {cfg.num_epochs}")
        logger.info(f"Model saved: {final_weights_path}")
        logger.info(f"Configuration saved: {os.path.join(save_dir, 'config_params.txt')}")

        # Optional: Save training history plot
        if not debug and len(history.history.get('loss', [])) > 0:
            try:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(10, 6))
                plt.plot(history.history['loss'], label='Training Loss')
                if 'val_loss' in history.history:
                    plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training History')
                plt.legend()
                plt.grid(True)

                plot_path = os.path.join(save_dir, 'training_history.png')
                plt.savefig(plot_path)
                plt.close()
                logger.info(f"Training history plot saved to: {plot_path}")
            except Exception as e:
                logger.warning(f"Could not save training plot: {e}")

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
    setup_logging(level=log_level)

    # Log startup information
    logger.info("=" * 60)
    logger.info("CONSTRAINED CONTRASTIVE LEARNING PRE-TRAINING")
    logger.info("=" * 60)
    logger.info(f"Debug mode: {args.debug}")
    logger.info(f"GPU ID: {args.gpu}")
    logger.info(f"Resume path: {args.resume}")

    # Run main training
    main(
        debug=args.debug,
        gpu_id=args.gpu,
        resume_path=args.resume
    )