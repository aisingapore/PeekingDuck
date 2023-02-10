"""
Loss functions Class to interact with Tensorflow

    @staticmethod
    def CategoricalCrossentropy(parameters):
        return tf.keras.losses.CategoricalCrossentropy(**parameters)

    @staticmethod
    def SparseCategoricalCrossentropy(parameters):
        return tf.keras.losses.SparseCategoricalCrossentropy(**parameters)

"""

from omegaconf import DictConfig
import tensorflow as tf

class TensorFlowLossAdapter:

    @staticmethod
    def get_loss_func(name, parameters: DictConfig = {}):
        return getattr(tf.keras.losses, name)(**parameters) if len(parameters) > 0 else getattr(tf.keras.losses, name)()


