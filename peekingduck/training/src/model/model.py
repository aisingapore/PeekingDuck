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

"""Model Interface that follows the Strategy Pattern."""
from __future__ import annotations

import os
import sys

sys.path.insert(1, os.getcwd())

import copy
import timm
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from omegaconf import DictConfig

from src.model.base import Model
from src.utils.general_utils import seed_all, rsetattr

# TODO: Follow timm's creation of head and backbone
class ImageClassificationModel(Model):
    """A generic image classification model. This is generic in the sense that
    it can be used for any image classification by just modifying the head.
    """

    def __init__(self, model_config: DictConfig) -> None:
        super().__init__(model_config)

        self.adapter = self.model_config.adapter  # leave out the adapter first
        self.model_name = self.model_config.model_name
        self.pretrained = self.model_config.pretrained
        self.model = self.create_model()

        # self.model.apply(self._init_weights) # activate if init weights
        print(f"Successfully created model: {self.model_config.model_name}")

    def _concat_backbone_and_head(self, last_layer_name) -> nn.Module:
        """Concatenate the backbone and head of the model."""
        model = copy.deepcopy(self.backbone)
        head = copy.deepcopy(self.head)
        rsetattr(model, last_layer_name, head)
        return model

    def create_model(self) -> nn.Module:
        """Create the model sequentially."""
        self.backbone = self.load_backbone()
        last_layer_name, _, in_features = self.get_last_layer()
        rsetattr(self.backbone, last_layer_name, nn.Identity())
        self.head = self.modify_head(in_features)
        model = self._concat_backbone_and_head(last_layer_name)
        return model

    def load_backbone(self) -> nn.Module:
        """Load the backbone of the model.

        NOTE:
            1. Backbones are usually loaded from timm or torchvision.
            2. This is not mandatory since users can just create it in create_model.
        """
        if self.adapter == "torchvision":
            backbone = getattr(torchvision.models, self.model_name)(
                pretrained=self.pretrained
            )
        elif self.adapter == "timm":
            backbone = timm.create_model(
                self.model_name,
                pretrained=self.pretrained,
                # in_chans=3
            )
        else:
            raise ValueError(f"Adapter {self.adapter} not supported.")
        return backbone

    def modify_head(self, in_features: int = None) -> nn.Module:
        """Modify the head of the model.

        NOTE/TODO:
            This part is very tricky, to modify the head,
            the penultimate layer of the backbone is taken, but different
            models will have different names for the penultimate layer.
        """
        # fully connected
        out_features = self.model_config.num_classes
        head = nn.Linear(in_features=in_features, out_features=out_features)
        return head

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model up to the penultimate layer to get
        feature embeddings.
        """
        features = self.backbone(inputs)
        return features

    def forward_head(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model up to the head to get predictions."""
        # nn.AdaptiveAvgPool2d(1)(inputs) is used by both timm and torchvision
        outputs = self.head(inputs)
        return outputs

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        features = self.forward_features(inputs)
        outputs = self.forward_head(features)
        return outputs
        # return self.model(inputs)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights of the model."""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.constant_(module.bias, 0)

    def forward_pass(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        y = self.model(inputs)
        print(f"X: {inputs.shape}, Y: {y.shape}")
        print("Forward Pass Successful")


# class MNISTModel(Model):
#     """MNIST Model."""

#     def __init__(self, pipeline_config: DictConfig) -> None:
#         super().__init__(pipeline_config)
#         self.pipeline_config = pipeline_config
#         self.create_model()

#     def create_model(self) -> MNISTModel:
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d(p=self.pipeline_config.model.dropout)
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, self.pipeline_config.model.num_classes)
#         return self

#     def forward(self, inputs: torch.Tensor) -> torch.Tensor:
#         inputs = F.relu(F.max_pool2d(self.conv1(inputs), 2))
#         inputs = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(inputs)), 2))
#         inputs = inputs.view(-1, 320)
#         inputs = F.relu(self.fc1(inputs))
#         inputs = F.dropout(inputs, training=self.training)
#         inputs = self.fc2(inputs)
#         return inputs


# if __name__ == "__main__":
#     seed_all(42)

#     pipeline_config = Cifar10PipelineConfig()

#     model = ImageClassificationModel(pipeline_config).to(pipeline_config.device)
#     print(model.model_summary(device=pipeline_config.device))

#     inputs = torch.randn(1, 3, 224, 224).to(pipeline_config.device)
#     model.forward_pass(inputs)
