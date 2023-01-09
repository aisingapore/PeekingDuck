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
from pathlib import Path
from typing import Optional, Tuple, Union
from omegaconf import DictConfig

import albumentations as A
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

TransformTypes = Optional[Union[A.Compose, T.Compose]]


class ImageClassificationDataset(Dataset):
    """A sample template for Image Classification Dataset."""

    def __init__(
        self,
        cfg: DictConfig,
        **kwargs,
    ) -> None:
        """"""

        super().__init__(**kwargs)
        self.cfg: DictConfig = cfg
        self.df = None

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.df)

    def __getitem__(
        self, index: int
    ) -> Union[torch.FloatTensor, Union[torch.FloatTensor, torch.LongTensor]]:
        """ """
        
    @classmethod
    def from_df(self, cfg: DictConfig,)
        """Creates an instance of the dataset class from a dataframe.
        """