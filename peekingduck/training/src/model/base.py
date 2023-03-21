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

"""Base class for Model."""

from abc import ABC, abstractmethod

import torch
from torch import nn

from configs.base import Config


class Model(ABC, nn.Module):
    """Model Base Class."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def create_backbone(self) -> nn.Module:
        """Creates the backbone of the model."""

    @abstractmethod
    def create_head(self) -> nn.Module:
        """Creates the head of the model."""

    @abstractmethod
    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of feature maps to get embeddings."""

    @abstractmethod
    def forward_head(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the head to get logits."""

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
