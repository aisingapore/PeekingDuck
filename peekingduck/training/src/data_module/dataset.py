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

from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


from src.data_module.base import DataModule


class ImageClassificationDataset(Dataset):
    """A sample template for Image Classification Dataset."""


class ImageClassificationDataModule(DataModule):
    """Data module for generic image classification dataset."""

    def __init__(
        self,
        cfg: DictConfig,
        **kwargs,
    ) -> None:
        """"""

        super().__init__(**kwargs)
