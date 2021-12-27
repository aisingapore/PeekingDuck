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
OSNet-AIN model architecture.
"""

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


# Basic layers
class LightConvStream(nn.Module):
    """Lightweight convolution stream."""

    def __init__(self, in_channels: int, out_channels: int, depth: int) -> None:
        super().__init__()
        assert depth >= 1, f"depth must be equal to or larger than 1, but got {depth}"
        layers = []
        layers += [LightConv3x3(in_channels, out_channels)]
        for _ in range(depth - 1):
            layers += [LightConv3x3(out_channels, out_channels)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return self.layers(x)


# Building blocks for omni-scale feature learning
class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(  # pylint: disable=unused-argument
        self,
        in_channels: int,
        out_channels: int,
        reduction: int = 4,
        num_streams: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        assert num_streams >= 1
        assert out_channels >= reduction and out_channels % reduction == 0
        mid_channels = out_channels // reduction

        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList()
        for t in range(1, num_streams + 1):
            self.conv2 += [LightConvStream(mid_channels, mid_channels, t)]
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2_t = conv2_t(x1)
            x2 = x2 + self.gate(x2_t)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.relu(out)


class OSBlockINin(nn.Module):
    """Omni-scale feature learning block with instance normalization."""

    def __init__(  # pylint: disable=unused-argument
        self,
        in_channels: int,
        out_channels: int,
        reduction: int = 4,
        num_streams: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        assert num_streams >= 1
        assert out_channels >= reduction and out_channels % reduction == 0
        mid_channels = out_channels // reduction

        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList()
        for t in range(1, num_streams + 1):
            self.conv2 += [LightConvStream(mid_channels, mid_channels, t)]
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels, bn=False)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2_t = conv2_t(x1)
            x2 = x2 + self.gate(x2_t)
        x3 = self.conv3(x2)
        x3 = self.IN(x3)  # IN inside residual
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
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
        conv1_instance_norm: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.loss = loss
        self.feature_dim = feature_dim

        # convolutional backbone
        self.conv1 = ConvLayer(
            3, channels[0], 7, stride=2, padding=3, instance_norm=conv1_instance_norm
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0], channels[1])
        self.pool2 = nn.Sequential(
            Conv1x1(channels[1], channels[1]), nn.AvgPool2d(2, stride=2)
        )
        self.conv3 = self._make_layer(blocks[1], layers[1], channels[1], channels[2])
        self.pool3 = nn.Sequential(
            Conv1x1(channels[2], channels[2]), nn.AvgPool2d(2, stride=2)
        )
        self.conv4 = self._make_layer(blocks[2], layers[2], channels[2], channels[3])
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # fully connected layer
        self.fc = self._construct_fc_layer(
            self.feature_dim, channels[3], dropout_p=None
        )
        # identity classification layer
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._init_params()

    def _make_layer(  # pylint: disable=unused-argument, no-self-use
        self, blocks: OSBlock, layer: int, in_channels: int, out_channels: int
    ) -> nn.Module:
        layers = []
        layers += [blocks[0](in_channels, out_channels)]  # type: ignore
        for i in range(1, len(blocks)):  # type: ignore
            layers += [blocks[i](out_channels, out_channels)]  # type: ignore
        return nn.Sequential(*layers)

    def _construct_fc_layer(
        self, fc_dims: int, input_dim: int, dropout_p: Optional[float] = None
    ) -> Optional[nn.Module]:
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None

        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]  # type: ignore

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))  # type: ignore
            layers.append(nn.ReLU())  # type: ignore
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))  # type: ignore
            input_dim = dim

        self.feature_dim = fc_dims[-1]  # type: ignore

        return nn.Sequential(*layers)

    def _init_params(self) -> None:
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

            elif isinstance(m, nn.InstanceNorm2d):
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
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(
        self, x: torch.Tensor, return_featuremaps: bool = False
    ) -> torch.Tensor:
        """Defines the computation performed at every call."""
        x = self.featuremaps(x)
        if return_featuremaps:
            return x
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        return y


# Instantiation
def osnet_ain_x1_0(
    num_classes: int = 1000, loss: str = "softmax", **kwargs: Any
) -> nn.Module:
    """Initializes osnet_x1_0 model with pretrained weights.

    Args:
        num_classes (int): Number of outputs for the classifier.
            Defaults to 1000.
        loss (str): Loss function used for evaluating model.
            Defaults to "softmax".

    Returns:
        nn.Module: OSNet-AIN model.
    """
    model = OSNet(
        num_classes,
        blocks=[
            [OSBlockINin, OSBlockINin],  # type: ignore
            [OSBlock, OSBlockINin],  # type: ignore
            [OSBlockINin, OSBlock],  # type: ignore
        ],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        loss=loss,
        conv1_IN=True,
        **kwargs,
    )

    return model
