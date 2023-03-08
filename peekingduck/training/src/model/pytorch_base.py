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

import functools
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
import torchinfo
from torch import nn
from torchinfo.model_statistics import ModelStatistics
from omegaconf import DictConfig


class PTModel(ABC, nn.Module):
    """Model Base Class."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.backbone: Optional[nn.Module]
        self.head: Optional[nn.Module]
        self.model: nn.Module
        self.model_config: DictConfig = cfg

    @abstractmethod
    def create_model(self) -> nn.Module:
        """Create the model.
        Note that users can implement anything they want, as long
        as the shape matches.
        """
        raise NotImplementedError("Please implement your own model.")

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""

    @abstractmethod
    def create_backbone(self) -> nn.Module:
        """Load the backbone of the model."""

    @abstractmethod
    def create_head(self, in_features: int) -> nn.Module:
        """Modify the head of the model."""

    def model_summary(
        self, input_size: Optional[Tuple[int, int, int, int]] = None, **kwargs: Any
    ) -> ModelStatistics:
        """Wrapper for torchinfo package to get the model summary."""
        if input_size is None:
            input_size = (
                1,
                3,
                self.model_config.augmentation.image_size,
                self.model_config.augmentation.image_size,
            )
        return torchinfo.summary(self.model, input_size=input_size, **kwargs)

    def get_last_layer(self) -> Tuple[Any, Optional[nn.Module], Any]:
        """Get the last layer information of a PyTorch Model.

        NOTE:
            Only correct if the last layer is a linear layer and is the head.
            Used by torchvision models.
            Does not work with timm's (e.g. vgg16) as the last layer is a
            flatten layer ('Identity()') with no 'in_features' attribute.
            (Only the second last layer is 'Linear')
        """
        # propagate through the model to get the last layer name
        for name, _ in getattr(self.backbone, "named_modules")():
            last_layer_name = name
        last_layer_attributes = last_layer_name.split(".")  # + ['in_features']
        # reduce applies to a list recursively and reduce it to a single value
        linear_layer = functools.reduce(getattr, last_layer_attributes, self.backbone)
        in_features = getattr(linear_layer, "in_features")
        last_layer_name = ".".join(last_layer_attributes)
        return last_layer_name, linear_layer, in_features
