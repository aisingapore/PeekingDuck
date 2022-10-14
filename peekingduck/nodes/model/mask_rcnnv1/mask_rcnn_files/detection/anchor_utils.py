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
#
# Original Copyright From PyTorch:
#
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America
#           (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute
#           (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
# From Caffe2:
#
# Copyright (c) 2016-present, Facebook Inc. All rights reserved.
#
# All contributions by Facebook:
# Copyright (c) 2016 Facebook Inc.
#
# All contributions by Google:
# Copyright (c) 2015 Google Inc.
# All rights reserved.
#
# All contributions by Yangqing Jia:
# Copyright (c) 2015 Yangqing Jia
# All rights reserved.
#
# All contributions by Kakao Brain:
# Copyright 2019-2020 Kakao Brain
#
# All contributions by Cruise LLC:
# Copyright (c) 2022 Cruise LLC.
# All rights reserved.
#
# All contributions from Caffe:
# Copyright(c) 2013, 2014, 2015, the respective contributors
# All rights reserved.
#
# All other contributions:
# Copyright(c) 2015, 2016 the respective contributors
# All rights reserved.
#
# Caffe2 uses a copyright model similar to Caffe: each contributor holds
# copyright over their contributions to Caffe2. The project versioning records
# all such contribution and copyright details. If a contributor wants to further
# mark their specific copyright on a particular contribution, they should
# indicate their copyright solely in the commit message of the change when it is
# committed.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Anchor Generator class for Faster-RCNN model

Modifications include:
- Removed conversion for sizes and aspect_ratios in __init__ of AnchorGenerator
"""

from typing import List, Tuple, Union
import torch
from torch import nn, Size, Tensor  # pylint: disable=no-name-in-module
from peekingduck.pipeline.nodes.model.mask_rcnnv1.mask_rcnn_files.detection.image_list import (
    ImageList,
)


class AnchorGenerator(nn.Module):
    """
    Module that generates anchors for a set of feature maps and
    image sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map. This module assumes aspect ratio = height / width for
    each anchor.

    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
    per spatial location for feature map i.

    Args:
        sizes (Tuple[Tuple[int, ...], ...]): The sizes of the generated anchor boxes
        aspect_ratios (Tuple[Tuple[float, ...], ...]): The aspect ratios of the generated anchor
            boxes
    """

    def __init__(
        self,
        sizes: Tuple[Tuple[int, ...], ...] = ((128, 256, 512),),
        aspect_ratios: Tuple[Tuple[float, ...], ...] = ((0.5, 1.0, 2.0),),
    ):
        super().__init__()

        assert isinstance(sizes[0], (list, tuple))
        assert isinstance(aspect_ratios[0], (list, tuple))
        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = [
            self.generate_anchors(size, aspect_ratio)
            for size, aspect_ratio in zip(sizes, aspect_ratios)
        ]

    @staticmethod
    def generate_anchors(
        scales: Union[List[int], Tuple[int, ...]],
        aspect_ratios: Union[List[float], Tuple[float, ...]],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        """Method to generate anchors"""
        scales_tensor = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios_tensor = torch.as_tensor(
            aspect_ratios, dtype=dtype, device=device
        )
        height_ratios = torch.sqrt(aspect_ratios_tensor)
        width_ratios = 1 / height_ratios

        width_scales = (width_ratios[:, None] * scales_tensor[None, :]).view(-1)
        height_scales = (height_ratios[:, None] * scales_tensor[None, :]).view(-1)

        base_anchors = (
            torch.stack(
                [-width_scales, -height_scales, width_scales, height_scales], dim=1
            )
            / 2
        )
        return base_anchors.round()

    def set_cell_anchors(self, dtype: torch.dtype, device: torch.device) -> None:
        """Set anchors to target device"""
        self.cell_anchors = [
            cell_anchor.to(dtype=dtype, device=device)
            for cell_anchor in self.cell_anchors
        ]

    def num_anchors_per_location(self) -> List[int]:
        """Generate a list of number of anchors per location"""
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def grid_anchors(
        self, grid_sizes: List[Size], strides: List[List[Tensor]]
    ) -> List[Tensor]:
        # pylint: disable=too-many-locals
        """For every combination of (a, (g, s), i) in (self.cell_anchors,
        zip(grid_sizes, strides), 0:2), output g[i] anchors that are s[i] distance
        apart in direction i, with the same dimensions as a.
        """
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None

        if not len(grid_sizes) == len(strides) == len(cell_anchors):
            raise ValueError(
                "Anchors should be Tuple[Tuple[int]] because each feature "
                "map could potentially have different sizes and aspect ratios. "
                "There needs to be a match between the number of "
                "feature maps passed and the number of sizes / aspect ratios specified."
            )

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            shifts_x = (
                torch.arange(0, grid_width, dtype=torch.int32, device=device)
                * stride_width
            )
            shifts_y = (
                torch.arange(0, grid_height, dtype=torch.int32, device=device)
                * stride_height
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def forward(
        self, image_list: ImageList, feature_maps: List[Tensor]
    ) -> List[Tensor]:
        """Generates anchors for a set of feature maps and image sizes.

        Args:
            image_list (ImageList):  An object that holds a list of padded images with its original
                image sizes
            feature_maps (List[Tensor]): List of feature maps for generating the anchors. Feature
                maps have the length equal to the number of feature levels, and the tensor of each
                element is in the shape of [batch_size, 256, feature_height, feature_width]. The
                number of channels 256 is based on the paper
                "Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>
                under the sub-section "Top-down pathway and lateral connections", where the number
                of channels of the features from the top-down pathway is fixed to d=256.

        Returns:
            List[Tensor]: List of generated anchors. The number of elements in the list corresponds
                to the batch size and in each element, there are 4 columns, each corresponds to the
                coordinates of the anchor boxes [x0, y0, x1, y1].
        """
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [
            [
                torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device),
            ]
            for g in grid_sizes
        ]
        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        anchors: List[List[Tensor]] = []
        for _ in range(len(image_list.image_sizes)):
            anchors.append(anchors_over_all_feature_maps)
        anchors_out = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        return anchors_out
