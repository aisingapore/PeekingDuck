"""
Scheduler Class to interact with Tensorflow

    @staticmethod
    def PiecewiseConstantDecay(parameters):
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(**parameters)

    @staticmethod
    def ExponentialDecay(parameters):
        return tf.keras.optimizers.schedules.ExponentialDecay(**parameters)

    @staticmethod
    def CosineDecay(parameters):
        return tf.keras.optimizers.schedules.CosineDecay(**parameters)

    @staticmethod
    def InverseTimeDecay(parameters):
        return tf.keras.optimizers.schedules.InverseTimeDecay(**parameters)

    @staticmethod
    def PolynomialDecay(parameters):
        return tf.keras.optimizers.schedules.PolynomialDecay(**parameters)

    @staticmethod
    def CosineDecayRestarts(parameters):
        return tf.keras.optimizers.schedules.CosineDecayRestarts(**parameters)

"""

from typing import Any, Dict
from omegaconf import DictConfig
import tensorflow as tf
import torch


class OptimizerSchedules:
    @staticmethod
    def get_tensorflow_scheduler(name, parameters: DictConfig = {}):
        return (
            getattr(tf.keras.optimizers.schedules, name)(**parameters)
            if len(parameters) > 0
            else getattr(tf.keras.optimizers.schedules, name)()
        )

    @staticmethod
    def get_pytorch_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler: str,
        parameters: DictConfig = {}
    ) -> torch.optim.lr_scheduler:
        """Get the scheduler for the optimizer."""
        return getattr(torch.optim.lr_scheduler, scheduler)(
            optimizer=optimizer, **parameters
        )
