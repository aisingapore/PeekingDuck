"""
    Optimizer Class Adapter for Tensorflow

    # @staticmethod
    # def Adadelta(learning_rate: LearningRateSchedule, parameters):
    #     raise NotImplementedError
    #     return tf.keras.optimizers.Adadelta(learning_rate=learning_rate, **parameters)

    # @staticmethod
    # def Adafactor(learning_rate: LearningRateSchedule, parameters):
    #     raise NotImplementedError
    #     return tf.keras.optimizers.Adafactor(learning_rate=learning_rate, **parameters)

    # @staticmethod
    # def Adagrad(learning_rate: LearningRateSchedule, parameters):
    #     raise NotImplementedError
    #     return tf.keras.optimizers.Adagrad(learning_rate=learning_rate, **parameters)

    @staticmethod
    def Adam(learning_rate: LearningRateSchedule, parameters):
        return .Adam(learning_rate=learning_rate, **parameters)

    # @staticmethod # This is not released yet for tf-macos==2.10
    # def AdamW(learning_rate: LearningRateSchedule, parameters):
    #     raise NotImplementedError
    #     # return tf.keras.optimizers.experimental.AdamW(learning_rate=learning_rate, **parameters)

    # @staticmethod
    # def Adamax(learning_rate: LearningRateSchedule, parameters):
    #     raise NotImplementedError
    #     return tf.keras.optimizers.Adamax(learning_rate=learning_rate, **parameters)

    # @staticmethod
    # def Ftrl(learning_rate: LearningRateSchedule, parameters):
    #     raise NotImplementedError
    #     return tf.keras.optimizers.Ftrl(learning_rate=learning_rate, **parameters)

    # @staticmethod
    # def Nadam(learning_rate: LearningRateSchedule, parameters):
    #     raise NotImplementedError
    #     return tf.keras.optimizers.Nadam(learning_rate=learning_rate, **parameters)

    # @staticmethod
    # def RMSprop(learning_rate: LearningRateSchedule, parameters):
    #     raise NotImplementedError
    #     return tf.keras.optimizers.RMSprop(learning_rate=learning_rate, **parameters)

    @staticmethod
    def SGD(learning_rate: LearningRateSchedule, parameters):
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, **parameters)


"""

import tensorflow as tf

from omegaconf import DictConfig


class OptimizersAdapter:

    @staticmethod
    def get_optimizer(name: str, learning_rate: tf.keras.optimizers.schedules.LearningRateSchedule, parameters: DictConfig = {}):
        return getattr(tf.keras.optimizers, name)(learning_rate, **parameters) if len(parameters) > 0 else getattr(tf.keras.optimizers, name)(learning_rate)
