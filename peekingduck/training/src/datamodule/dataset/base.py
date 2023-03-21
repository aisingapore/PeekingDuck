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

"""Base class for Dataset."""

from abc import ABC, abstractmethod
from typing import Union

import torch
from torch.utils.data import Dataset

from configs.base import Config


class ImageDataset(ABC, Dataset):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def __len__(self) -> int:
        """Returns the length of the dataset."""

    @abstractmethod
    def __getitem__(
        self, index
    ) -> Union[torch.FloatTensor, Union[torch.FloatTensor, torch.LongTensor]]:
        """Gets the item at index."""
