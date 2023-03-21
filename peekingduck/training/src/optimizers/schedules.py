# Copyright 2023 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

from omegaconf import DictConfig
import tensorflow as tf
import torch


class OptimizerSchedules:
    """Optimizer Schedules"""

    @staticmethod
    def get_tensorflow_scheduler(
        name: str, parameters: DictConfig
    ) -> tf.keras.optimizers.schedules.LearningRateSchedule:
        """get_tensorflow_scheduler"""
        return (
            getattr(tf.keras.optimizers.schedules, name)(**parameters)
            if len(parameters) > 0
            else getattr(tf.keras.optimizers.schedules, name)()
        )

    @staticmethod
    def get_pytorch_scheduler(
        optimizer: torch.optim.Optimizer, scheduler: str, parameters: DictConfig
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Get the scheduler for the optimizer."""
        return getattr(torch.optim.lr_scheduler, scheduler)(
            optimizer=optimizer, **parameters
        )
