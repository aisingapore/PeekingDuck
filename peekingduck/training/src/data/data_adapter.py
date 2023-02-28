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

from typing import Any, Union
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from albumentations import Compose

from src.data.base import AbstractDataSet
from src.data.dataset import TFImageClassificationDataset


class DataAdapter:
    """Adapter for different framework"""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg: DictConfig = cfg
        self.loader: None = None
        if cfg.adapter_type == "pytorch":
            self.loader = DataLoader
        if cfg.adapter_type == "tensorflow":
            self.loader = TFImageClassificationDataset

    def train_dataloader(
        self, dataset: AbstractDataSet, transforms: Compose
    ) -> Union[Any, None]:
        if self.cfg.adapter_type == "pytorch":
            return self.loader(
                dataset,
                **self.cfg.train,
            )
        if self.cfg.adapter_type == "tensorflow":
            return self.loader(
                dataset,
                transforms=transforms,
                **self.cfg.train,
            )

    def validation_dataloader(
        self, dataset: AbstractDataSet, transforms: Compose
    ) -> Union[Any, None]:
        if self.cfg.adapter_type == "pytorch":
            return self.loader(
                dataset,
                **self.cfg.valid,
            )
        if self.cfg.adapter_type == "tensorflow":
            return self.loader(
                dataset,
                transforms=transforms,
                **self.cfg.valid,
            )

    def test_dataloader(
        self, dataset: AbstractDataSet, transforms: Compose
    ) -> Union[Any, None]:
        if self.cfg.adapter_type == "pytorch":
            return self.loader(
                dataset,
                **self.cfg.test,
            )
        if self.cfg.adapter_type == "tensorflow":
            return self.loader(
                dataset,
                transforms=transforms,
                **self.cfg.test,
            )
