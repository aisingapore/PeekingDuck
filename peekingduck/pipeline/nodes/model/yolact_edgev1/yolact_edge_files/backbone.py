# Modifications copyright 2022 AI Singapore
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

# MIT License

# Copyright (c) 2020 Haotian Liu and Rafael A. Rivera-Soto

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Backbone for YolactEdge.
Modifications include:
- Removed unused ResNetBackboneGN class
- Removed unused darknetconvlayer
- Removed unused DarkNetBlock class
- Removed unused DarkNetBackbone class
- Removed unused VGGBackBone class
- Refactor and formatting
"""

from typing import Callable, List, Any, Tuple, Union
from functools import partial
from torch import nn
from torch import Tensor


class Bottleneck(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Adapted from torchvision.models.resnet"""

    expansion = 4

    def __init__(  # pylint: disable=too-many-arguments
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Callable = None,
        norm_layer: Callable = nn.BatchNorm2d,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, bias=False, dilation=dilation
        )
        self.bn1 = norm_layer(planes)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = norm_layer(planes)

        self.conv3 = nn.Conv2d(
            planes, planes * 4, kernel_size=1, bias=False, dilation=dilation
        )
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        if downsample is not None:
            self.downsample = downsample
        else:
            self.downsample = nn.Sequential()
        self.stride = stride

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward propagation of the Bottleneck blocks of ResNet.

        Args:
            inputs (Tensor): Input tensor

        Returns:
            out (Tensor): Output tensor
        """
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.downsample(inputs)
        out += residual
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Implements ResNet backbone, adapted from torchvision.models.resnet"""

    def __init__(
        self,
        layers: List[int],
        block: Callable = Bottleneck,
        norm_layer: Callable = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels: List[Any] = []
        self.norm_layer = norm_layer
        self.dilation = 1
        self.atrous_layers: List[Any] = []

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._make_layer(block, 64, layers[0])
        self._make_layer(block, 128, layers[1], stride=2)
        self._make_layer(block, 256, layers[2], stride=2)
        self._make_layer(block, 512, layers[3], stride=2)

        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    def _make_layer(
        self, block: Callable, planes: int, blocks: int, stride: int = 1
    ) -> Callable:
        """Method to make layers for ResNet"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if len(self.layers) in self.atrous_layers:
                self.dilation += 1
                stride = 1

            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    dilation=self.dilation,
                ),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.norm_layer,
                self.dilation,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self.norm_layer))

        layer = nn.Sequential(*layers)
        self.channels.append(planes * block.expansion)
        self.layers.append(layer)
        return layer

    def forward(self, inputs: Tensor, partial_bn: bool = False) -> List[Tensor]:
        """Returns a list of convouts for each layer."""
        inputs = self.conv1(inputs)
        inputs = self.bn1(inputs)
        inputs = self.relu(inputs)
        inputs = self.maxpool(inputs)

        outs = []
        layer_idx = 0
        for layer in self.layers:
            layer_idx += 1
            if not partial_bn or layer_idx <= 2:
                inputs = layer(inputs)
                outs.append(inputs)
        return outs

    def add_layer(
        self,
        conv_channels: int = 1024,
        downsample: int = 2,
        depth: int = 1,
        block: Callable = Bottleneck,
    ) -> None:
        """Add a downsample layer to the backbone as per what SSD does."""
        self._make_layer(
            block, conv_channels // block.expansion, blocks=depth, stride=downsample
        )


class ConvBNAct(nn.Sequential):
    """Adapted from torchvision.models.mobilenet.ConvBNReLU"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: Any = nn.ReLU6(inplace=True),
    ) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            activation,
        )


ConvBNReLU = partial(ConvBNAct)


class InvertedResidual(nn.Module):
    """Adapted from torchvision.models.mobilenet.InvertedResidual"""

    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[Union[ConvBNAct, nn.Conv2d, nn.BatchNorm2d]] = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

        layers.extend(
            [
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """Returns a convout for the inverted residual block."""
        if self.use_res_connect:
            return inputs + self.conv(inputs)
        return self.conv(inputs)


class MobileNetV2Backbone(nn.Module):
    """Adapted from torchvision.models.mobilenet.MobileNetV2"""

    def __init__(
        self,
        width_mult: float = 1.0,
        inverted_residual_setting: Any = None,
        round_nearest: int = 8,
        block: Callable = InvertedResidual,
    ) -> None:
        super().__init__()

        input_channel = 32
        last_channel = 1280
        self.channels = []
        self.layers = nn.ModuleList()

        if inverted_residual_setting is None:
            raise ValueError(
                "Must provide inverted_residual_setting where each "
                "element is a list that represents the "
                "MobileNetV2 t,c,n,s values for that layer."
            )
        if (
            len(inverted_residual_setting) == 0
            or len(inverted_residual_setting[0]) != 4
        ):
            raise ValueError(
                "inverted_residual_setting should be non-empty "
                "or a 4-element list, got {}".format(inverted_residual_setting)
            )

        input_channel = self._make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = self._make_divisible(
            last_channel * max(1.0, width_mult), round_nearest
        )
        self.layers.append(ConvBNReLU(3, input_channel, stride=2))
        self.channels.append(input_channel)

        # t_val: expansion factor
        # c_val: output channels
        # n_val: number of repetitions
        # s_val: stride of the first layer of each sequence
        for t_val, c_val, n_val, s_val in inverted_residual_setting:
            input_channel = self._make_layer(
                input_channel,
                width_mult,
                round_nearest,
                t_val,
                c_val,
                n_val,
                s_val,
                block,
            )
            self.channels.append(input_channel)

        self.layers.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.channels.append(self.last_channel)
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    def _make_layer(  # pylint: disable=too-many-arguments
        self,
        input_channel: int,
        width_mult: float,
        round_nearest: int,
        t_val: int,
        c_val: int,
        n_val: int,
        s_val: int,
        block: Callable,
    ) -> int:
        """A layer is a combination of inverted residual blocks"""
        layers = []
        output_channel = self._make_divisible(c_val * width_mult, round_nearest)

        for i in range(n_val):
            stride = s_val if i == 0 else 1
            layers.append(
                block(input_channel, output_channel, stride, expand_ratio=t_val)
            )
            input_channel = output_channel

        self.layers.append(nn.Sequential(*layers))
        return input_channel

    def forward(self, inputs: Tensor) -> Tuple[Tensor, ...]:
        """Returns a list of convouts for each layer."""
        outs = []

        for _, layer in enumerate(self.layers):
            inputs = layer(inputs)
            outs.append(inputs)

        return tuple(outs)

    def add_layer(  # pylint: disable=too-many-arguments
        self,
        conv_channels: int = 1280,
        t_val: int = 1,
        c_val: int = 1280,
        n_val: int = 1,
        s_val: int = 2,
    ) -> None:
        """
        Args:
            conv_channels: number of channels in the convout of the previous layer
            t_val: expansion factor
            c_val: output channels
            n_val: number of repetitions
            s_val: stride of the first layer of each sequence
        """
        self._make_layer(
            conv_channels, 1.0, 8, t_val, c_val, n_val, s_val, InvertedResidual
        )

    @classmethod
    def _make_divisible(  # pylint: disable=bad-classmethod-argument
        self, value: float, divisor: int, min_value: Any = None
    ) -> int:
        """Adapted from torchvision.models.mobilenet._make_divisable"""
        if min_value is None:
            min_value = divisor
        new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_value < 0.9 * value:
            new_value += divisor
        return new_value
