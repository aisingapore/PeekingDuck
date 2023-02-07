"""
Loss functions Class to interact with Tensorflow

"""

import tensorflow as tf

class TensorFlowLossAdapter:

    @staticmethod
    def CategoricalCrossentropy(parameters):
        return tf.keras.losses.CategoricalCrossentropy(**parameters)

    @staticmethod
    def SparseCategoricalCrossentropy(parameters):
        return tf.keras.losses.SparseCategoricalCrossentropy(**parameters)