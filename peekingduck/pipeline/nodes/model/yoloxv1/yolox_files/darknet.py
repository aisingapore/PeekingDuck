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

"""Backbone for YOLOPAFPN.

Modifications include:
- Removed unused Darknet class
- Removed unused DWConv, ResLayer import
- Removed depthwise and act arguments
- Refactor and formatting
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from peekingduck.pipeline.nodes.model.yoloxv1.yolox_files.network_blocks import (
    BaseConv,
    CSPLayer,
    Focus,
    SPPBottleneck,
)


class CSPDarknet(nn.Module):
    """Modified CSPNet with SiLU activation"""

    # pylint: disable=arguments-differ
    def __init__(
        self,
        dep_mul: float,
        wid_mul: float,
        out_features: Tuple[str, str, str],
    ) -> None:
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features

        channels = int(wid_mul * 64)  # 64
        depth = max(round(dep_mul * 3), 1)  # 3

        self.stem = Focus(3, channels, 3)
        self.dark2 = nn.Sequential(*self.make_group_layer(channels, depth))
        self.dark3 = nn.Sequential(*self.make_group_layer(channels * 2, depth * 3))
        self.dark4 = nn.Sequential(*self.make_group_layer(channels * 4, depth * 3))
        self.dark5 = nn.Sequential(
            BaseConv(channels * 8, channels * 16, 3, 2),
            SPPBottleneck(channels * 16, channels * 16),
            CSPLayer(channels * 16, channels * 16, depth, False),
        )

    @staticmethod
    def make_group_layer(in_channels: int, depth: int) -> Tuple[BaseConv, CSPLayer]:
        """Starts with BaseConv layer, followed by a CSPLayer"""
        return (
            BaseConv(in_channels, in_channels * 2, 3, 2),
            CSPLayer(in_channels * 2, in_channels * 2, depth),
        )

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Defines the computation performed at every call"""
        outputs = {}
        inputs = self.stem(inputs)
        outputs["stem"] = inputs
        inputs = self.dark2(inputs)
        outputs["dark2"] = inputs
        inputs = self.dark3(inputs)
        outputs["dark3"] = inputs
        inputs = self.dark4(inputs)
        outputs["dark4"] = inputs
        inputs = self.dark5(inputs)
        outputs["dark5"] = inputs
        return {k: v for k, v in outputs.items() if k in self.out_features}
