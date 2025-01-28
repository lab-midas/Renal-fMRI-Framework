import  tensorflow as tf
import os, logging
from wandb.integration.keras import WandbCallback
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger


def setup_TF_environment(gpus_available):
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpus_available
    # Suppressing TF message printouts
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
    logging.getLogger('tensorflow').setLevel(logging.FATAL)
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    return

class CustomCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, save_every_epochs=1, save_initial_model=True):
        super(CustomCheckpointCallback, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.save_every_epochs = save_every_epochs
        self.save_initial_model = save_initial_model

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_every_epochs == 0 or (epoch == 0 and self.save_initial_model):
            # Create a checkpoint directory if it doesn't exist
            os.makedirs(self.checkpoint_dir, exist_ok=True)

            # Generate a unique file path with the epoch number
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"model_epoch_{epoch + 1:04d}.h5"
            )

            # Save the model checkpoint
            self.model.save_weights(checkpoint_path)

            print(f"Saved checkpoint at {checkpoint_path}")


def get_callbacks(csvPath, checkpoint_dir):
    ckptPath = os.path.join(checkpoint_dir,'checkpoints')
    os.makedirs(ckptPath, exist_ok=True)
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                factor=0.8,
                                patience=10,
                                verbose=1),
                 CSVLogger(csvPath),
                 WandbCallback(),
                 CustomCheckpointCallback(ckptPath, save_every_epochs=10)
    ]
    return callbacks


class CustomCheckpointCallbackSaveBest(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, save_every_epochs=1, save_initial_model=True):
        super(CustomCheckpointCallbackSaveBest, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.save_every_epochs = save_every_epochs
        self.save_initial_model = save_initial_model
        self.best_val_loss = float('inf')  # Initialize best validation loss to infinity

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')  # Get current validation loss from logs
        if val_loss is not None:
            # Check if the current validation loss is better than the best seen so far
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss  # Update the best validation loss
                # Create a checkpoint directory if it doesn't exist
                os.makedirs(self.checkpoint_dir, exist_ok=True)

                # Save the model weights to a specific file
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, f"best_model_epoch_{epoch + 1:04d}_val_dsc_{val_loss:.3f}.h5"
                )

                self.model.save_weights(checkpoint_path)  # Save the weights
                print(f"Saved best model at epoch {epoch + 1} with val_loss: {val_loss:.4f} at {checkpoint_path}")

        if (epoch + 1) % self.save_every_epochs == 0 or (epoch == 0 and self.save_initial_model):
            # Create a checkpoint directory if it doesn't exist
            os.makedirs(self.checkpoint_dir, exist_ok=True)

            # Generate a unique file path with the epoch number
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"model_epoch_{epoch + 1:04d}.h5"
            )

            # Save the model checkpoint
            self.model.save_weights(checkpoint_path)

            print(f"Saved checkpoint at {checkpoint_path}")

def get_callbacks_save_best(csvPath, checkpoint_dir):
    ckptPath = os.path.join(checkpoint_dir,'checkpoints')
    os.makedirs(ckptPath, exist_ok=True)
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                factor=0.8,
                                patience=10,
                                verbose=1),
                 CSVLogger(csvPath),
                 WandbCallback(),
                 CustomCheckpointCallbackSaveBest(ckptPath, save_every_epochs=10)
    ]
    return callbacks