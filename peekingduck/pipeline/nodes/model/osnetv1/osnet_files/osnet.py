# Copyright 2021 AI Singapore
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
#
# Original copyright (c) 2018 Kaiyang Zhou
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
OSNet model architecture.
"""

from __future__ import division, absolute_import
from typing import Any, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
from peekingduck.pipeline.nodes.model.osnetv1.osnet_files.network_blocks import (
    ConvLayer,
    Conv1x1,
    Conv1x1Linear,
    LightConv3x3,
    ChannelGate,
)


# pylint: disable=invalid-name


# Building blocks for omni-scale feature learning
class OSBlock(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Omni-scale feature learning block."""

    def __init__(  # pylint: disable=unused-argument
        self,
        in_channels: int,
        out_channels: int,
        IN: bool = False,
        bottleneck_reduction: int = 4,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            IN (bool): Applies Instance Normalization. Defaults to False.
            bottleneck_reduction (int): Reduction rate in the bottleneck
                architecture. Defaults to 4.
        """
        super().__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


# Network architecture
class OSNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Omni-Scale Network."""

    def __init__(  # pylint: disable=too-many-arguments, unused-argument
        self,
        num_classes: int,
        blocks: List[OSBlock],
        layers: List[int],
        channels: List[int],
        feature_dim: int = 512,
        loss: str = "softmax",
        IN: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            num_classes (int): Classes number/person ids number.
            blocks (List[OSBlock]): OSBlock form basic block for OSNet.
            layers (List[int]): Stores the stack number for OSBlock stack.
            channels (List[int]): Stores the channel number for channel regulation.
            feature_dim (int): Fully connected output vector size.
                Defaults to 512.
            loss (str): String stores loss type for experiment loss
                regulation. Defaults to "softmax".
            IN (bool): Applies Instance Normalization. Defaults to False.
        """
        super().__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.loss = loss
        self.feature_dim = feature_dim

        # convolutional backbone
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(
            blocks[0],
            layers[0],
            channels[0],
            channels[1],
            reduce_spatial_size=True,
            IN=IN,
        )
        self.conv3 = self._make_layer(
            blocks[1], layers[1], channels[1], channels[2], reduce_spatial_size=True
        )
        self.conv4 = self._make_layer(
            blocks[2], layers[2], channels[2], channels[3], reduce_spatial_size=False
        )
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # fully connected layer
        self.fc = self._construct_fc_layer(
            self.feature_dim, channels[3], dropout_p=None
        )
        # identity classification layer
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._init_params()

    def _make_layer(  # pylint: disable=too-many-arguments, no-self-use
        self,
        block: OSBlock,
        layer: int,
        in_channels: int,
        out_channels: int,
        reduce_spatial_size: bool,
        IN: bool = False,
    ) -> nn.Module:
        """Creates a layer block."""
        layers = []

        layers.append(block(in_channels, out_channels, IN=IN))
        for _ in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN))

        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels), nn.AvgPool2d(2, stride=2)
                )
            )

        return nn.Sequential(*layers)

    def _construct_fc_layer(
        self, fc_dims: int, input_dim: int, dropout_p: Optional[float] = None
    ) -> Optional[nn.Module]:
        """Constructs fully connected layer."""
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None

        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]  # type: ignore

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))  # type: ignore
            layers.append(nn.BatchNorm1d(dim))  # type: ignore
            layers.append(nn.ReLU(inplace=True))  # type: ignore
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))  # type: ignore
            input_dim = dim

        self.feature_dim = fc_dims[-1]  # type: ignore

        return nn.Sequential(*layers)

    def _init_params(self) -> None:
        """Initializes parameters for network."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x: torch.Tensor) -> torch.Tensor:
        """Parses input through network layers."""
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x  # Returns features

    def forward(
        self, x: torch.Tensor, return_featuremaps: bool = False
    ) -> torch.Tensor:
        """Defines the computation performed at every call."""
        x = self.featuremaps(x)
        if return_featuremaps:
            return x  # Returns features
        v = self.global_avgpool(x)
        # Create view tensor for fast and memory efficient operations
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        return y


# Instantiation
def osnet_x1_0(
    num_classes: int = 1000, loss: str = "softmax", **kwargs: Any
) -> nn.Module:
    """Initializes osnet_x1_0 model with pretrained weights.

    Args:
        num_classes (int): Number of outputs for the classifier.
            Defaults to 1000.
        loss (str): Loss function used for evaluating model.
            Defaults to "softmax".

    Returns:
        nn.Module: OSNet model.
    """
    # Standard size (width x1.0)
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],  # type: ignore
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        loss=loss,
        **kwargs,
    )

    return model
