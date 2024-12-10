import tensorflow as tf
from src.utils.custom_losses import DIRECTION_WEIGHT_FACTOR

class DirectionWeightCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs: int):
        super().__init__()
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        # Start factor at 3.0 and linearly move towards 1.0 by the end of training
        phase_ratio = epoch / float(self.total_epochs)
        factor = 3.0 - 2.0 * phase_ratio
        if factor < 1.0:
            factor = 1.0
        DIRECTION_WEIGHT_FACTOR.assign(factor)
        print(f"Epoch {epoch+1}: Direction Weight Factor = {factor:.2f}")
