#!/usr/bin/env python
"""
Utility functions for multi-parametric renal MRI preprocessing and analysis.

This module provides core utility functions for:
- Logging configuration
- TensorFlow environment setup
- Training callbacks (checkpointing, logging, learning rate scheduling)
- Image processing (contrast stretching, cropping, normalization, denoising)
- Mask conversion utilities
- Metric computation (Dice score)

All functions are designed to work with numpy arrays representing medical images,
typically with dimensions (height, width, depth) or (height, width, depth, channels).
"""

import os
import sys
import logging
from typing import Optional, List, Tuple, Union, Dict, Any

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, Callback


try:
    from wandb.integration.keras import WandbCallback

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging(
        level: Union[int, str] = logging.INFO,
        log_file: Optional[str] = None,
        console_output: bool = True,
        log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> None:
    """
    Setup logging configuration for the application.

    This function configures the root logger with consistent formatting
    and optional file output.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG, or string "INFO")
        log_file: Optional path to log file. If provided, logs will be written to file.
        console_output: Whether to output logs to console
        log_format: Format string for log messages

    Example:
        >>> setup_logging(level=logging.DEBUG, log_file='training.log')
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Training started")
    """
    # Convert string level to logging constant if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplication
    root_logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Add file handler if log file specified
    if log_file:
        # Create directory for log file if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Suppress verbose logging from TensorFlow
    tf.get_logger().setLevel(logging.ERROR)

    # Suppress TensorFlow CPP logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error

    # Log startup message
    logging.info(f"Logging configured: level={logging.getLevelName(level)}, file={log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    This is a convenience wrapper around logging.getLogger() that ensures
    the logger is properly configured.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Loading data...")
    """
    return logging.getLogger(name)


# =============================================================================
# TENSORFLOW ENVIRONMENT SETUP
# =============================================================================

def setup_TF_environment(
        gpus_available: Optional[str] = None,
        memory_growth: bool = True,
        suppress_logs: bool = True
) -> None:
    """
    Setup TensorFlow environment with appropriate GPU settings.

    This function configures GPU memory growth and visibility to prevent
    TensorFlow from allocating all GPU memory at once.

    Args:
        gpus_available: Comma-separated string of GPU IDs to use (e.g., '0,1')
                       If None, uses all available GPUs.
        memory_growth: Whether to enable memory growth (prevents TF from
                      allocating all GPU memory at once)
        suppress_logs: Whether to suppress TensorFlow info/warning logs

    Example:
        >>> setup_TF_environment('0,1')  # Use GPUs 0 and 1
    """
    logger = get_logger(__name__)

    # Suppress TensorFlow logs if requested
    if suppress_logs:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL/ERROR only
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

    # Set visible GPUs if specified
    if gpus_available is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus_available
        logger.info(f"Setting visible GPUs: {gpus_available}")

    # Configure GPU memory growth
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                if memory_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)

            logical_gpus = tf.config.list_logical_devices('GPU')
            logger.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
            if memory_growth:
                logger.info("Enabled memory growth for all GPUs")
        else:
            logger.warning("No GPUs found, running on CPU")

    except RuntimeError as e:
        logger.error(f"Error configuring GPUs: {e}")
        logger.warning("Continuing with default GPU configuration")


# =============================================================================
# CUSTOM CALLBACKS
# =============================================================================

class ModelCheckpointCallback(Callback):
    """
    Custom callback to save model checkpoints at regular intervals.

    This callback saves model weights every N epochs and optionally saves
    the initial model.

    Attributes:
        checkpoint_dir: Directory to save checkpoints
        save_every_epochs: Save checkpoint every N epochs
        save_initial_model: Whether to save the model at epoch 0
    """

    def __init__(
            self,
            checkpoint_dir: str,
            save_every_epochs: int = 1,
            save_initial_model: bool = True
    ):
        """
        Initialize the checkpoint callback.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_every_epochs: Save checkpoint every N epochs
            save_initial_model: Whether to save the model at epoch 0
        """
        super(ModelCheckpointCallback, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.save_every_epochs = save_every_epochs
        self.save_initial_model = save_initial_model
        self.logger = get_logger(__name__)

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.logger.info(f"Checkpoint directory: {checkpoint_dir}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Save checkpoint at epoch end if conditions are met.

        Args:
            epoch: Current epoch number (0-indexed)
            logs: Training metrics dictionary
        """
        epoch_num = epoch + 1  # Convert to 1-indexed for user-friendly naming

        # Check if we should save at this epoch
        should_save = (
                (epoch_num % self.save_every_epochs == 0) or
                (epoch == 0 and self.save_initial_model)
        )

        if should_save:
            # Generate checkpoint filename
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"model_epoch_{epoch_num:04d}.h5"
            )

            # Save model weights
            self.model.save_weights(checkpoint_path)
            self.logger.info(f"Saved checkpoint at epoch {epoch_num}: {checkpoint_path}")


class BestModelCheckpointCallback(Callback):
    """
    Custom callback to save the best model based on validation loss.

    This callback saves model weights whenever validation loss improves,
    and also saves regular checkpoints every N epochs.

    Attributes:
        checkpoint_dir: Directory to save checkpoints
        save_every_epochs: Save regular checkpoint every N epochs
        save_initial_model: Whether to save the model at epoch 0
        monitor: Metric to monitor for best model (default: 'val_loss')
        mode: 'min' or 'max' - whether to minimize or maximize the monitored metric
    """

    def __init__(
            self,
            checkpoint_dir: str,
            save_every_epochs: int = 1,
            save_initial_model: bool = True,
            monitor: str = 'val_loss',
            mode: str = 'min'
    ):
        """
        Initialize the best model checkpoint callback.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_every_epochs: Save regular checkpoint every N epochs
            save_initial_model: Whether to save the model at epoch 0
            monitor: Metric to monitor for best model
            mode: 'min' or 'max' - whether to minimize or maximize the monitored metric
        """
        super(BestModelCheckpointCallback, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.save_every_epochs = save_every_epochs
        self.save_initial_model = save_initial_model
        self.monitor = monitor
        self.mode = mode

        # Initialize best value based on mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.logger = get_logger(__name__)

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.logger.info(f"Checkpoint directory: {checkpoint_dir}")
        self.logger.info(f"Monitoring: {monitor} (mode: {mode})")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Save checkpoints at epoch end.

        Args:
            epoch: Current epoch number (0-indexed)
            logs: Training metrics dictionary
        """
        if logs is None:
            logs = {}

        epoch_num = epoch + 1  # Convert to 1-indexed

        # Check if we should save best model
        current_value = logs.get(self.monitor)
        if current_value is not None:
            improved = False

            if self.mode == 'min' and current_value < self.best_value:
                improved = True
                self.best_value = current_value
            elif self.mode == 'max' and current_value > self.best_value:
                improved = True
                self.best_value = current_value

            if improved:
                # Save best model
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"best_model_epoch_{epoch_num:04d}_{self.monitor}_{current_value:.4f}.h5"
                )
                self.model.save_weights(checkpoint_path)
                self.logger.info(
                    f"Saved best model at epoch {epoch_num} with {self.monitor}: {current_value:.4f}"
                )

        # Save regular checkpoints
        should_save = (
                (epoch_num % self.save_every_epochs == 0) or
                (epoch == 0 and self.save_initial_model)
        )

        if should_save and (epoch_num % self.save_every_epochs == 0 or epoch == 0):
            # Generate regular checkpoint filename
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"model_epoch_{epoch_num:04d}.h5"
            )
            self.model.save_weights(checkpoint_path)
            self.logger.info(f"Saved regular checkpoint at epoch {epoch_num}")


# =============================================================================
# CALLBACK FACTORY FUNCTIONS
# =============================================================================

def get_callbacks(
        csv_path: str,
        checkpoint_dir: str,
        use_wandb: bool = True,
        reduce_lr_factor: float = 0.8,
        reduce_lr_patience: int = 10,
        save_every_epochs: int = 10,
        monitor: str = 'val_loss'
) -> List[Callback]:
    """
    Create a list of training callbacks for pre-training.

    Args:
        csv_path: Path to save CSV log file
        checkpoint_dir: Directory to save model checkpoints
        use_wandb: Whether to use Weights & Biases callback
        reduce_lr_factor: Factor to reduce learning rate by
        reduce_lr_patience: Patience for learning rate reduction
        save_every_epochs: Save checkpoint every N epochs
        monitor: Metric to monitor for learning rate reduction

    Returns:
        List of Keras callbacks

    Example:
        >>> callbacks = get_callbacks('logs.csv', 'checkpoints/')
        >>> model.fit(x, y, callbacks=callbacks)
    """
    logger = get_logger(__name__)
    callbacks = []

    # Create checkpoint subdirectory
    ckpt_path = os.path.join(checkpoint_dir, 'checkpoints')
    os.makedirs(ckpt_path, exist_ok=True)

    # 1. Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor,
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        verbose=1,
        mode='min',
        min_delta=1e-4,
        cooldown=0,
        min_lr=1e-7
    )
    callbacks.append(reduce_lr)
    logger.debug(f"Added ReduceLROnPlateau (factor={reduce_lr_factor}, patience={reduce_lr_patience})")

    # 2. CSV logger
    csv_logger = CSVLogger(csv_path, separator=',', append=True)
    callbacks.append(csv_logger)
    logger.debug(f"Added CSVLogger: {csv_path}")

    # 3. Weights & Biases callback (if available and requested)
    if use_wandb and WANDB_AVAILABLE:
        try:
            wandb_callback = WandbCallback(
                save_model=False,
                log_model=True,
                log_graph=True
            )
            callbacks.append(wandb_callback)
            logger.debug("Added WandbCallback")
        except Exception as e:
            logger.warning(f"Failed to initialize WandbCallback: {e}")
    elif use_wandb and not WANDB_AVAILABLE:
        logger.warning("WandB not installed. Install with: pip install wandb")

    # 4. Custom checkpoint callback
    checkpoint_callback = ModelCheckpointCallback(
        ckpt_path,
        save_every_epochs=save_every_epochs
    )
    callbacks.append(checkpoint_callback)
    logger.debug(f"Added ModelCheckpointCallback (every {save_every_epochs} epochs)")

    logger.info(f"Created {len(callbacks)} training callbacks")
    return callbacks


def get_callbacks_save_best(
        csv_path: str,
        checkpoint_dir: str,
        use_wandb: bool = True,
        reduce_lr_factor: float = 0.8,
        reduce_lr_patience: int = 10,
        save_every_epochs: int = 50,
        monitor: str = 'val_loss',
        mode: str = 'min'
) -> List[Callback]:
    """
    Create a list of training callbacks that save the best model.

    This version includes a callback that saves the model whenever the
    monitored metric improves.

    Args:
        csv_path: Path to save CSV log file
        checkpoint_dir: Directory to save model checkpoints
        use_wandb: Whether to use Weights & Biases callback
        reduce_lr_factor: Factor to reduce learning rate by
        reduce_lr_patience: Patience for learning rate reduction
        save_every_epochs: Save regular checkpoint every N epochs
        monitor: Metric to monitor for best model and learning rate
        mode: 'min' or 'max' - whether to minimize or maximize the monitored metric

    Returns:
        List of Keras callbacks

    Example:
        >>> callbacks = get_callbacks_save_best('logs.csv', 'checkpoints/')
        >>> model.fit(x, y, callbacks=callbacks)
    """
    logger = get_logger(__name__)
    callbacks = []

    # Create checkpoint subdirectory
    ckpt_path = os.path.join(checkpoint_dir, 'checkpoints')
    os.makedirs(ckpt_path, exist_ok=True)

    # 1. Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor,
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        verbose=1,
        mode=mode,
        min_delta=1e-4,
        cooldown=0,
        min_lr=1e-7
    )
    callbacks.append(reduce_lr)
    logger.debug(f"Added ReduceLROnPlateau (factor={reduce_lr_factor}, patience={reduce_lr_patience})")

    # 2. CSV logger
    csv_logger = CSVLogger(csv_path, separator=',', append=True)
    callbacks.append(csv_logger)
    logger.debug(f"Added CSVLogger: {csv_path}")

    # 3. Weights & Biases callback (if available and requested)
    if use_wandb and WANDB_AVAILABLE:
        try:
            wandb_callback = WandbCallback(
                save_model=False,
                log_model=True,
                log_graph=True
            )
            callbacks.append(wandb_callback)
            logger.debug("Added WandbCallback")
        except Exception as e:
            logger.warning(f"Failed to initialize WandbCallback: {e}")
    elif use_wandb and not WANDB_AVAILABLE:
        logger.warning("WandB not installed. Install with: pip install wandb")

    # 4. Best model checkpoint callback
    best_checkpoint_callback = BestModelCheckpointCallback(
        ckpt_path,
        save_every_epochs=save_every_epochs,
        monitor=monitor,
        mode=mode
    )
    callbacks.append(best_checkpoint_callback)
    logger.debug(f"Added BestModelCheckpointCallback (every {save_every_epochs} epochs)")

    logger.info(f"Created {len(callbacks)} training callbacks (with best model saving)")
    return callbacks

