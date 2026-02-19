#!/usr/bin/env python
"""
Non-Rigid (Deformable) Registration Training Script

This script trains a deformable registration network to align multiple MRI contrasts
to a template contrast. It builds upon a pre-trained affine registration model
for initialization and then learns non-rigid deformations.

The network predicts dense displacement fields that warp moving images to the
fixed template space, with regularization to ensure smooth deformations.

Usage:
    python train_reg_nonrigid.py [--debug] [--gpu GPU_ID] [--resume CHECKPOINT]

Example:
    python train_reg_nonrigid.py --debug --gpu 3
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import wandb
import tensorflow as tf
from tensorflow.keras import optimizers

# Add project root to path to ensure imports work
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Local imports
import configs.registration_nonrigid as cfg
from data.train_reg_Data_Generator import DataLoaderRegistration
from losses.reg_losses import residuce_loss, Grad, MutualInformation2, DiceLoss2Classes
from models.model_reg_gn import modelObj
from utils import setup_TF_environment, get_callbacks, setup_logging

# Configure logging
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Non-Rigid Registration Training')

    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode (limited data, no callbacks)')

    parser.add_argument('--gpu', type=str, default='2',
                        help='GPU ID to use (e.g., "0" or "0,1" for multiple)')

    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')

    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Override experiment name from config')

    parser.add_argument('--no_affine', action='store_true',
                        help='Disable affine pre-training initialization')

    return parser.parse_args()


def setup_wandb(config, debug):
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
            project="Renfi_registration",
            entity=config.wandb_entity,
            group="nonrigid",
            name=config.experiment_name,
            config={
                'batch_size': config.batch_size,
                'learning_rate': config.lr_pretrain,
                'num_epochs': config.num_epochs,
                'template_contrast': config.contrast_template,
                'moving_contrasts': config.contrast_moving,
                'weighted': config.weighted,
                'affine_init': config.checkpoint_affine is not None
            },
            settings=wandb.Settings(start_method='thread')
        )
        logger.info(f"WandB initialized for experiment: {config.experiment_name}")
    except Exception as e:
        logger.error(f"Failed to initialize WandB: {e}")
        logger.warning("Continuing without WandB tracking...")


def create_model(config, no_affine=False):
    """
    Create non-rigid registration model with optional affine initialization.

    Args:
        config: Configuration object
        no_affine: If True, disable affine pre-training initialization

    Returns:
        Keras model
    """
    logger.info("Creating non-rigid registration model")

    # Override affine checkpoint if requested
    checkpoint_affine = None if no_affine else config.checkpoint_affine

    if checkpoint_affine:
        logger.info(f"Using affine pre-training from: {checkpoint_affine}")
    else:
        logger.info("Training without affine initialization")

    # Initialize model utility class
    mm_utils = modelObj(config)

    # Create groupwise non-rigid registration model with affine initialization
    model = mm_utils.reg_groupwise_affine_nonrigid(
        checkpoint_affine=checkpoint_affine,
        weighted=config.weighted
    )

    # Print model summary
    model.summary()

    return model


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
    train_gen = DataLoaderRegistration(config, debug=debug, train_flag=True)
    logger.info(f"Training batches: {len(train_gen)}")
    logger.info(f"Training samples: {train_gen.get_len()}")

    # Validation data loader
    val_gen = DataLoaderRegistration(config, debug=debug, train_flag=False)
    logger.info(f"Validation batches: {len(val_gen)}")
    logger.info(f"Validation samples: {val_gen.get_len()}")

    # Debug: test a single batch
    if debug:
        logger.info("Debug mode: Testing data loader...")
        try:
            inputs, targets = next(iter(train_gen))
            logger.info(f"Input shapes: {[i.shape for i in inputs]}")
            logger.info(f"Target shapes: {[t.shape for t in targets]}")

            # Test model prediction
            model = create_model(config, no_affine=False)
            outputs = model.predict(inputs)
            logger.info(f"Output shapes: {[o.shape for o in outputs]}")

        except Exception as e:
            logger.error(f"Error testing data loader: {e}")

    return train_gen, val_gen


def setup_loss_and_optimizer(config):
    """
    Setup loss functions and optimizer for non-rigid registration.

    Returns:
        Tuple of (loss_list, loss_weights, optimizer)
    """
    logger.info("Setting up loss functions and optimizer...")

    # Define loss functions for multi-output model
    # Output order: [warped_template, pred_fix_lbl, flow, ...]
    custom_loss = [
        MutualInformation2().loss,  # Image similarity loss (MI)
        DiceLoss2Classes().loss,  # Mask overlap loss
        Grad(penalty='l1').loss,  # Smoothness regularization on flow field
        residuce_loss  # Residual loss for consistency
    ]

    # Loss weights (tuned for optimal performance)
    loss_weights = [0.01, 1, 0.001, 10]

    logger.info(f"Loss weights:")
    logger.info(f"  • Mutual Information: {loss_weights[0]}")
    logger.info(f"  • Dice: {loss_weights[1]}")
    logger.info(f"  • Smoothness (L1): {loss_weights[2]}")
    logger.info(f"  • Residual: {loss_weights[3]}")

    # Create optimizer with gradient clipping for stability
    optimizer = optimizers.Adam(
        learning_rate=config.lr_pretrain,
        clipvalue=1.0
    )

    logger.info(f"Using Adam optimizer with learning rate: {config.lr_pretrain}")

    return custom_loss, loss_weights, optimizer


def setup_output_directories(config, debug):
    """
    Create output directories for saving models and logs.

    Args:
        config: Configuration object
        debug: Whether in debug mode

    Returns:
        Tuple of (save_dir, csv_path)
    """
    # Create experiment-specific save directory
    save_dir = os.path.join(config.base_save_dir, 'groupwise', config.experiment_name)

    if debug:
        logger.info(f"Debug mode: Would save to {save_dir}")
        return save_dir, None

    os.makedirs(save_dir, exist_ok=True)

    # Path for training log CSV
    csv_path = os.path.join(save_dir, 'training.log')

    logger.info(f"Output directory: {save_dir}")
    logger.info(f"Log file: {csv_path}")

    return save_dir, csv_path


def save_config(config, save_dir):
    """
    Save configuration to a text file.

    Args:
        config: Configuration object
        save_dir: Directory to save config file
    """
    config_path = os.path.join(save_dir, 'nonrigid_config_params.txt')

    with open(config_path, 'w') as f:
        # Write header
        f.write("# Non-Rigid Registration Configuration\n")
        f.write(f"# Generated: {__import__('datetime').datetime.now()}\n")
        f.write("#" + "=" * 60 + "\n\n")

        # Write all config attributes
        for name, value in sorted(config.__dict__.items()):
            # Skip private attributes and modules
            if not name.startswith('_') and not callable(value):
                f.write(f"{name} = {repr(value)}\n")

    logger.info(f"Configuration saved to: {config_path}")


def main(debug=False, gpu_id="2", resume_path=None, experiment_name=None, no_affine=False):
    """
    Main training function.

    Args:
        debug: Run in debug mode
        gpu_id: GPU ID to use
        resume_path: Path to checkpoint to resume from
        experiment_name: Override experiment name
        no_affine: Disable affine pre-training initialization
    """
    # Setup GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    logger.info(f"Using GPU(s): {gpu_id}")

    # Override configuration if specified
    if experiment_name:
        cfg.experiment_name = experiment_name
        logger.info(f"Overriding experiment name: {experiment_name}")

    if resume_path:
        cfg.checkpoint_path = resume_path
        logger.info(f"Resuming from: {resume_path}")

    # Store debug flag in config
    cfg.debug = debug

    # Setup TensorFlow environment
    setup_TF_environment(cfg.gpus_available)

    # Setup experiment tracking (WandB)
    setup_wandb(cfg, debug)

    try:
        # Step 1: Create model
        logger.info("=" * 60)
        logger.info("STEP 1: Creating model")
        logger.info("=" * 60)
        model = create_model(cfg, no_affine=no_affine)

        # Load checkpoint if provided
        if cfg.checkpoint_path and os.path.exists(cfg.checkpoint_path):
            logger.info(f"Loading weights from: {cfg.checkpoint_path}")
            model.load_weights(cfg.checkpoint_path, by_name=True)

        # Step 2: Create data loaders
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Creating data loaders")
        logger.info("=" * 60)
        train_gen, val_gen = create_data_loaders(cfg, debug)

        # Step 3: Setup loss and optimizer
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Setting up loss and optimizer")
        logger.info("=" * 60)
        loss_fns, loss_weights, optimizer = setup_loss_and_optimizer(cfg)

        # Compile model with multiple losses
        model.compile(
            optimizer=optimizer,
            loss=loss_fns,
            loss_weights=loss_weights,
            metrics={'pred_fix_lbl': DiceLoss2Classes().loss}
        )

        # Step 4: Setup output directories
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Setting up output directories")
        logger.info("=" * 60)
        save_dir, csv_path = setup_output_directories(cfg, debug)

        # Step 5: Setup callbacks
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: Setting up callbacks")
        logger.info("=" * 60)

        if debug:
            callbacks = []
            logger.info("Debug mode: No callbacks will be used")
        else:
            callbacks = get_callbacks(
                csv_path=csv_path,
                checkpoint_dir=save_dir,
                save_every_epochs=10
            )
            logger.info(f"Callbacks configured: {[c.__class__.__name__ for c in callbacks]}")

        # Step 6: Train model
        logger.info("\n" + "=" * 60)
        logger.info("STEP 6: Starting training")
        logger.info("=" * 60)
        logger.info(f"Epochs: {cfg.initial_epoch} to {cfg.num_epochs}")
        logger.info(f"Batch size: {cfg.batch_size}")
        logger.info(f"Steps per epoch: 1000")
        logger.info(f"Validation steps: 200")

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=cfg.num_epochs,
            steps_per_epoch=1000 if not debug else 10,
            validation_steps=200 if not debug else 2,
            verbose=1,
            workers=6,
            callbacks=callbacks,
            initial_epoch=cfg.initial_epoch,
            use_multiprocessing=False,
            shuffle=True
        )

        # Step 7: Save final model
        if not debug:
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
            if history.history.get('val_loss'):
                best_val_loss = min(history.history['val_loss'])
                best_epoch = history.history['val_loss'].index(best_val_loss) + 1
                logger.info(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

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
    logger.info("NON-RIGID REGISTRATION TRAINING")
    logger.info("=" * 60)
    logger.info(f"Debug mode: {args.debug}")
    logger.info(f"GPU ID: {args.gpu}")
    logger.info(f"Resume path: {args.resume}")
    logger.info(f"Experiment name override: {args.experiment_name}")
    logger.info(f"Disable affine init: {args.no_affine}")

    # Run main training
    main(
        debug=args.debug,
        gpu_id=args.gpu,
        resume_path=args.resume,
        experiment_name=args.experiment_name,
        no_affine=args.no_affine
    )