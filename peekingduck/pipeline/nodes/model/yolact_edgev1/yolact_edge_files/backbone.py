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

from typing import List
import torch.nn as nn
from functools import partial


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(
        self, 
        inplanes: int, 
        planes: int, 
        stride: int = 1, 
        downsample: nn.Module = None,
        norm_layer=nn.BatchNorm2d, 
        dilation: int=1
    ) -> None:
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, bias=False, dilation=dilation
        )
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)

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

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNetBackbone(nn.Module):
    def __init__(self, 
                 layers: List[int], 
                 atrous_layers: List[int] = [], 
                 block = Bottleneck, 
                 norm_layer = nn.BatchNorm2d):
        super().__init__()
        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels = []
        self.norm_layer = norm_layer
        self.dilation = 1
        self.atrous_layers = atrous_layers

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
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if len(self.layers) in self.atrous_layers:
                self.dilation += 1
                stride = 1
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False,
                          dilation=self.dilation),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, 
                            planes, 
                            stride, 
                            downsample,
                            self.norm_layer, 
                            self.dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self.norm_layer))

        layer = nn.Sequential(*layers)
        self.channels.append(planes * block.expansion)
        self.layers.append(layer)

        return layer

    def forward(self, x, partial:bool=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        layer_idx = 0
        for layer in self.layers:
            layer_idx += 1
            if not partial or layer_idx <= 2:
                x = layer(x)
                outs.append(x)
        return outs

    def add_layer(self, conv_channels=1024, downsample=2, depth=1, block=Bottleneck):
        self._make_layer(block, conv_channels // block.expansion, blocks=depth, stride=downsample)

# This function is necessary for MobileNetV2 implementation                
def _make_divisible(v, divisor, min_value=None):
    """
    Adapted from torchvision.models.mobilenet._make_divisable
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNAct(nn.Sequential):
    def __init__(
        self, 
        in_planes: int, 
        out_planes: int, 
        kernel_size: int = 3, 
        stride=1, 
        groups=1, 
        activation=nn.ReLU6(inplace=True)
    ) -> None:
        padding = (kernel_size - 1) // 2
        super(ConvBNAct, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, 
                groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            activation
        )

ConvBNReLU = partial(ConvBNAct)


class InvertedResidual(nn.Module):
    """
    Adapted from torchvision.models.mobilenet.InvertedResidual
    """
    def __init__(
        self, 
        inp: int, 
        oup: int, 
        stride: int, 
        expand_ratio
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2Backbone(nn.Module):
    """
    Adapted from torchvision.models.mobilenet.MobileNetV2
    """
    def __init__(self,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=InvertedResidual):
        super(MobileNetV2Backbone, self).__init__()

        input_channel = 32
        last_channel = 1280
        self.channels = []
        self.layers = nn.ModuleList()

        if inverted_residual_setting is None:
            raise ValueError("Must provide inverted_residual_setting where each element is a list "
                             "that represents the MobileNetV2 t,c,n,s values for that layer.")
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.layers.append(ConvBNReLU(3, input_channel, stride=2))
        self.channels.append(input_channel)

        for t, c, n, s in inverted_residual_setting:
            input_channel = self._make_layer(input_channel, width_mult, round_nearest, t, c, n, s, block)
            self.channels.append(input_channel)

        self.layers.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.channels.append(self.last_channel)
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    def _make_layer(self, input_channel, width_mult, round_nearest, t, c, n, s, block):
        layers = []
        output_channel = _make_divisible(c * width_mult, round_nearest)

        for i in range(n):
            stride = s if i == 0 else 1
            layers.append(block(input_channel, output_channel, stride, expand_ratio=t))
            input_channel = output_channel

        self.layers.append(nn.Sequential(*layers))
        return input_channel

    def forward(self, x):
        outs = []

        for idx, layer in enumerate(self.layers):
            x = layer(x)
            outs.append(x)
        
        return tuple(outs)

    def add_layer(self, conv_channels=1280, t=1, c=1280, n=1, s=2):
        self._make_layer(conv_channels, 1.0, 8, t, c, n, s, InvertedResidual)

# This function can be abstracted away since it is only called once
def construct_backbone():
    """Constructs a backbone given the respective backbone configuration."""
    backbone = ResNetBackbone(([3, 4, 23, 3])) # R101
    # backbone = ResNetBackbone(([3, 4, 6, 3])) # R50
    num_layers = max(list(range(1, 4))) + 1

    """MobileNetV2"""
    # backbone = MobileNetV2Backbone(1.0, [[1, 16, 1, 1], 
    #                                     [6, 24, 2, 2],
    #                                     [6, 32, 3, 2], 
    #                                     [6, 64, 4, 2], 
    #                                     [6, 96, 3, 1], 
    #                                     [6, 160, 3, 2], 
    #                                     [6, 320, 1, 1]], 8) # MobileNetV2
    # num_layers = max([3, 4, 6]) + 1 # MovileNetV2

    while len(backbone.layers) < num_layers:
        backbone.add_layer()
    return backbone