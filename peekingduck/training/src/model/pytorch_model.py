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

import logging
import copy
import timm
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from omegaconf import DictConfig

from src.model.pytorch_base import PTModel
from src.utils.general_utils import seed_all, rsetattr
from src.utils.pt_model_utils import set_trainable_layers

from configs import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name

# TODO: Follow timm's creation of head and backbone
class PTClassificationModel(PTModel):
    """A generic image classification model. This is generic in the sense that
    it can be used for any image classification by just modifying the head.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.adapter = self.model_config.adapter
        self.model_name = self.model_config.model_name
        self.pretrained = self.model_config.pretrained
        self.weights = self.model_config.weights
        self.unfreeze: str = self.model_config.unfreeze
        self.fine_tune_modules: DictConfig = self.model_config.fine_tune_modules
        self.model = self.create_model()
        logger.info(f"Successfully created model: {self.model_config.model_name}")

    def _concat_backbone_and_head(self, backbone, head, last_layer_name) -> nn.Module:
        """Concatenate the backbone and head of the model."""
        # model = copy.deepcopy(self.backbone)
        # head = copy.deepcopy(self.head)
        rsetattr(backbone, last_layer_name, head)
        return backbone

    def create_model(self) -> nn.Module:
        """Create the model sequentially."""

        self.backbone = self.load_backbone()

        if self.adapter == "torchvision":
            last_layer_name, _, in_features = self.get_last_layer()
            logger.info(
                f"last_layer_name: {last_layer_name}. in_features: {in_features}"
            )
            rsetattr(self.backbone, last_layer_name, nn.Identity())
            # create the head
            head = self.modify_head(in_features)
            # attach the head
            model = self._concat_backbone_and_head(self.backbone, head, last_layer_name)

        elif self.adapter == "timm":
            self.backbone.reset_classifier(num_classes=self.model_config.num_classes)
            model = self.backbone
        else:
            raise ValueError(f"Adapter {self.adapter} not supported.")

        # self.backbone = self.load_backbone()

        # if self.adapter == "torchvision":
        #     last_layer_name, _, in_features = self.get_last_layer()
        #     logger.info(
        #         f"last_layer_name: {last_layer_name}. in_features: {in_features}"
        #     )
        #     rsetattr(self.backbone, last_layer_name, nn.Identity())
        #     self.head = self.modify_head(in_features)
        #     model = self._concat_backbone_and_head(last_layer_name)
        # elif self.adapter == "timm":
        #     model = copy.deepcopy(self.backbone)
        #     model.reset_classifier(num_classes=self.model_config.num_classes)
        # show the available modules to unfreeze
        logger.info(
            f"Available modules to be unfroze are {[module for module in self.backbone._modules]}"
        )

        set_trainable_layers(model, self.model_config.fine_tune_modules)

        # unfreeze the model parameters based on the config
        # if self.unfreeze == "none":
        #     pass
        # elif self.unfreeze == "all":
        #     self.unfreeze_all_params(model)
        # elif self.unfreeze == "partial":
        #     self.unfreeze_partial_params(model, self.unfreeze_modules)
        # else:
        #     raise ValueError(f"Unfreeze setting '{self.unfreeze}' is not supported")

        return model

    def load_backbone(self) -> nn.Module:
        """Load the backbone of the model.

        NOTE:
            1. Backbones are usually loaded from timm or torchvision.
            2. This is not mandatory since users can just create it in create_model.
        """
        if self.adapter == "torchvision":
            backbone = getattr(torchvision.models, self.model_name)(
                weights=self.weights
            )
        elif self.adapter == "timm":
            backbone = timm.create_model(self.model_name, pretrained=self.pretrained)
        else:
            raise ValueError(f"Adapter {self.adapter} not supported.")
        # freeze the backbone by default
        self.freeze_all_params(backbone)
        return backbone

    def modify_head(self, in_features: int = None) -> nn.Module:
        """Modify the head of the model."""
        # fully connected
        out_features = self.model_config.num_classes
        head = nn.Linear(in_features=in_features, out_features=out_features)
        return head

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model up to the penultimate layer to get
        feature embeddings. Only works for torchvision backbone.
        """
        features = self.backbone(inputs)
        return features

    def forward_head(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model up to the head to get predictions.
        Only works for torchvision backbone.
        """
        # nn.AdaptiveAvgPool2d(1)(inputs) is used by both timm and torchvision
        outputs = self.head(inputs)
        return outputs

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model based on the adapter"""
        # if self.adapter == "torchvision":
        #     features = self.forward_features(inputs)
        #     outputs = self.forward_head(features)
        # elif self.adapter == "timm":
        #     outputs = self.model(inputs)
        # else:
        #     raise ValueError(f"Adapter {self.adapter} not supported.")

        outputs = self.model(inputs)

        return outputs
