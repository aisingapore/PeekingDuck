# Modifications copyright 2021 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Original copyright (c) 2019 ZhongdaoWang
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

"""Network blocks for constructing the Darknet-53 backbone of the JDE model.

Modifications include:
- Removed custom Upsample module
- Removed EmptyLayer module (the equivalent nn.Identity is available)
- Removed training related code in YOLOLayer.forward()
- Removed loss related member variables
- Removed img_size in constructor since it's ignored and self.img_size is
    initialised to 0 by default
- Removed `layer` member variable in YOLOLayer since it's not used
- Updating self.img_size after creating grid to avoid recreation
    - Refactored initial value of self.img_size to None to indicate it's
        meant to be overwritten
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from peekingduck.pipeline.nodes.model.jdev1.jde_files.utils import decode_delta_map


class YOLOLayer(nn.Module):  # pylint: disable=too-many-instance-attributes
    """YOLO detection layer.

    Args:
        anchors (List[Tuple[float, float]]): List of anchor box width and
            heights.
        num_classes (int): Number of classes. Uses 1 for JDE.
        num_identities (int): Number of identities, e.g., number of unique
            pedestrians. Uses 14455 for JDE according to the original code.
        embedding_dim (int): Size of embedding. Uses 512 for JDE according to
            the original code.
        device (torch.device): The device which a `torch.Tensor` is on or
            will be allocated.
    """

    def __init__(  # pylint:disable=too-many-arguments
        self,
        anchors: List[Tuple[float, float]],
        num_classes: int,
        num_identities: int,
        embedding_dim: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.anchors = torch.FloatTensor(anchors)
        self.num_anchors = len(anchors)  # number of anchors (4)
        self.num_classes = num_classes  # number of classes (80)
        self.num_identities = num_identities
        self.emb_dim = embedding_dim
        self.device = device
        self.img_size: Optional[Tuple[int, int]] = None
        self.shift = [1, 3, 5]

        self.anchor_vec: torch.Tensor
        self.anchor_wh: torch.Tensor
        self.grid_xy: torch.Tensor
        self.stride: float

        self.emb_scale = (
            math.sqrt(2) * math.log(self.num_identities - 1)
            if self.num_identities > 1
            else 1
        )

    def forward(self, inputs: torch.Tensor, img_size: Tuple[int, int]) -> torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            inputs (torch.Tensor): Feature maps at various scales.
            img_size (Tuple[int, int]): Image size as specified by backbone
                configuration.

        Returns:
            (torch.Tensor): A decoded tensor containing the prediction.
        """
        # From arxiv article:
        # Prediction map has dimension B * (6A + D) * H * W where A is number
        # of anchor templates, D is embedding dimension. B, H, and W are
        # batch size, height, and width (of the feature maps) respectively.
        pred_anchor, pred_embedding = inputs[:, :24, ...], inputs[:, 24:, ...]
        batch_size, grid_height, grid_width = (
            pred_anchor.shape[0],
            pred_anchor.shape[-2],
            pred_anchor.shape[-1],
        )

        if self.img_size != img_size:
            self._create_grids(img_size, grid_height, grid_width)
            # Only have to create the grid once for each image size
            self.img_size = img_size
            if pred_anchor.is_cuda:
                self.grid_xy = self.grid_xy.cuda()
                self.anchor_wh = self.anchor_wh.cuda()

        # prediction
        pred_anchor = (
            pred_anchor.view(
                batch_size,
                self.num_anchors,
                self.num_classes + 5,
                grid_height,
                grid_width,
            )
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        pred_embedding = pred_embedding.permute(0, 2, 3, 1).contiguous()
        pred_box = pred_anchor[..., :4]
        pred_conf = pred_anchor[..., 4:6].permute(0, 4, 1, 2, 3)

        pred_conf = torch.softmax(pred_conf, dim=1)[:, 1, ...].unsqueeze(-1)
        pred_embedding = F.normalize(
            pred_embedding.unsqueeze(1)
            .repeat(1, self.num_anchors, 1, 1, 1)
            .contiguous(),
            dim=-1,
        )
        pred_cls = torch.zeros(
            batch_size, self.num_anchors, grid_height, grid_width, 1
        ).to(self.device)
        pred_anchor = torch.cat([pred_box, pred_conf, pred_cls, pred_embedding], dim=-1)
        pred_anchor[..., :4] = decode_delta_map(
            pred_anchor[..., :4], self.anchor_vec.to(pred_anchor), self.device
        )
        pred_anchor[..., :4] *= self.stride

        return pred_anchor.view(batch_size, -1, pred_anchor.shape[-1])

    def _create_grids(
        self, img_size: Tuple[int, int], grid_height: int, grid_width: int
    ) -> None:
        """Builds the grid for anchor box offsets.

        Args:
            img_size (Tuple[int, int]): Model input size.
            grid_height (int): Height of grid.
            grid_width (int): Width of grid.
        """
        self.stride = img_size[0] / grid_width
        assert (
            self.stride == img_size[1] / grid_height
        ), f"Inconsistent stride size: {self.stride} v.s. {img_size[1]} / {grid_height}"

        # build xy offsets
        grid_x = (
            torch.arange(grid_width)
            .repeat((grid_height, 1))
            .view((1, 1, grid_height, grid_width))
            .float()
        )
        grid_y = (
            torch.arange(grid_height)
            .repeat((grid_width, 1))
            .transpose(0, 1)
            .view((1, 1, grid_height, grid_width))
            .float()
        )
        self.grid_xy = torch.stack((grid_x, grid_y), 4)

        # build wh gains
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.num_anchors, 1, 1, 2)
