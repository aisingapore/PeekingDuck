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

"""Data adapter class for dataloader"""

from dataclasses import dataclass, field
from typing import Any, Union
from omegaconf import DictConfig

import pandas as pd
from torch.utils.data import DataLoader
from albumentations import Compose

from src.data.base import AbstractDataSet
from src.data.dataset import TFImageClassificationDataset


@dataclass
class DataAdapter:
    """Adapter for different framework"""

    cfg: DictConfig
    loader: Any = field(init=False)

    def __post_init__(self) -> None:
        """__post_init__"""
        assert self.cfg.adapter_type in [
            "pytorch",
            "tensorflow",
        ], f"Unsupported framework {self.cfg.adapter_type}"

        if self.cfg.adapter_type == "pytorch":
            self.loader = DataLoader
        if self.cfg.adapter_type == "tensorflow":
            self.loader = TFImageClassificationDataset

    def train_dataloader(
        self, dataset: Union[AbstractDataSet, pd.DataFrame], transforms: Compose
    ) -> Union[DataLoader, AbstractDataSet]:
        """train_dataloader"""
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

        raise UnboundLocalError(f"Unsupported framework {self.cfg.adapter_type}")

    def validation_dataloader(
        self, dataset: Union[AbstractDataSet, pd.DataFrame], transforms: Compose
    ) -> Union[DataLoader, AbstractDataSet]:
        """validation_dataloader"""
        if self.cfg.adapter_type == "pytorch":
            return self.loader(
                dataset,
                **self.cfg.validation,
            )
        if self.cfg.adapter_type == "tensorflow":
            return self.loader(
                dataset,
                transforms=transforms,
                **self.cfg.validation,
            )

        raise UnboundLocalError(f"Unsupported framework {self.cfg.adapter_type}")

    def test_dataloader(
        self, dataset: Union[AbstractDataSet, pd.DataFrame], transforms: Compose
    ) -> Union[DataLoader, AbstractDataSet]:
        """test_dataloader"""
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

        raise UnboundLocalError(f"Unsupported framework {self.cfg.adapter_type}")
