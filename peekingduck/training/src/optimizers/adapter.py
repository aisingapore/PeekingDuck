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
    Optimizer Class Adapter for Tensorflow
"""
from typing import Any, Dict

from omegaconf import DictConfig
import torch
import tensorflow as tf


class OptimizersAdapter:
    """Tensorflow Optimizers Adapter"""

    @staticmethod
    def get_tensorflow_optimizer(
        name: str,
        learning_rate: tf.keras.optimizers.schedules.LearningRateSchedule,
        parameters: DictConfig,
    ) -> tf.keras.optimizers.Optimizer:
        """get_tensorflow_optimizer"""
        return (
            getattr(tf.keras.optimizers, name)(learning_rate, **parameters)
            if len(parameters) > 0
            else getattr(tf.keras.optimizers, name)(learning_rate)
        )

    @staticmethod
    def get_pytorch_optimizer(
        model: torch.nn.Module,
        optimizer: str,
        optimizer_params: Dict[str, Any],
    ) -> torch.optim.Optimizer:
        """Get the pytorch optimizer for the model.
        Note:
            Do not invoke self.model directly in this call as it may affect model initalization.
            https://stackoverflow.com/questions/70107044/can-i-define-a-method-as-an-attribute
        """
        return getattr(torch.optim, optimizer)(
            filter(lambda p: p.requires_grad, model.parameters()), **optimizer_params
        )
