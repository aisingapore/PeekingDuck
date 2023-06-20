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

"""Trainer class protocol"""

from typing import Any, Dict, Protocol, Union
from omegaconf import DictConfig

from torch.utils.data import DataLoader

from src.data.base import AbstractDataSet


# pylint: disable=too-many-arguments
class Trainer(Protocol):
    """Object used to facilitate training."""

    def setup(
        self,
        trainer_config: DictConfig,
        model_config: DictConfig,
        callbacks_config: DictConfig,
        metrics_config: DictConfig,
        data_config: DictConfig,
        device: str = "",
    ) -> None:
        """Setup"""

    def train(
        self,
        train_loader: Union[DataLoader, AbstractDataSet],
        validation_loader: Union[DataLoader, AbstractDataSet],
    ) -> Dict[str, Any]:
        """Trainer train"""
        ...
