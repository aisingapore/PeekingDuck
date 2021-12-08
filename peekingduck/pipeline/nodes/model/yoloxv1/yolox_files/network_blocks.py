# Modifications copyright 2021 AI Singapore
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
# Original copyright 2021 Megvii, Base Detection
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Network blocks for constructing the YOLOX model.

Modifications include:
- BaseConv
    - Removed get_activations, uses SiLU only
    - Removed groups and bias arguments, uses group=1 and bias=False for
        nn.Conv2d only
- CSPLayer
    - Removed expansion, uses 0.5 only
- Remove SiLU export-friendly class
- Removed unused DWConv and ResLayer class
- Removed depthwise and act arguments
- Refactor and formatting
"""

from typing import Tuple

import torch
import torch.nn as nn


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> SiLU block."""

    # pylint: disable=invalid-name
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int,
    ) -> None:
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride, pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return self.act(self.bn(self.conv(inputs)))

    def fuseforward(self, inputs: torch.Tensor) -> torch.Tensor:
        """The computation performed at every call when conv and batch norm
        layers are fused.
        """
        return self.act(self.conv(inputs))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, 1)
        self.conv2 = BaseConv(hidden_channels, out_channels, 3, 1)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        outputs = self.conv2(self.conv1(inputs))
        if self.use_add:
            outputs = outputs + inputs
        return outputs


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP."""

    # pylint: disable=invalid-name
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Tuple[int, int, int] = (5, 9, 13),
    ) -> None:
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(ks, 1, ks // 2) for ks in kernel_sizes])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        inputs = self.conv1(inputs)
        inputs = torch.cat([inputs] + [m(inputs) for m in self.m], dim=1)
        inputs = self.conv2(inputs)
        return inputs


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions."""

    # pylint: disable=invalid-name
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        shortcut: bool = True,
    ) -> None:
        """
        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            num_blocks (int): Number of Bottlenecks, default = 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * 0.5)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, 1)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, 1)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, 1)
        self.m = nn.Sequential(
            *[
                Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        inputs_1 = self.conv1(inputs)
        inputs_2 = self.conv2(inputs)
        inputs_1 = self.m(inputs_1)
        inputs = torch.cat((inputs_1, inputs_2), dim=1)
        return self.conv3(inputs)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    # pylint: disable=invalid-name
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = inputs[..., ::2, ::2]
        patch_top_right = inputs[..., ::2, 1::2]
        patch_bot_left = inputs[..., 1::2, ::2]
        patch_bot_right = inputs[..., 1::2, 1::2]
        inputs = torch.cat(
            (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1
        )
        return self.conv(inputs)
