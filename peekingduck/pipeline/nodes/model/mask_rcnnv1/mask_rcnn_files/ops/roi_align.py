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

"""RoiAlign function for the MultiScaleRoIAlign class"""

from typing import List, Union

import torch
from torch import Tensor
from torch.nn.modules.utils import _pair
from torch.jit.annotations import BroadcastingList2
from torchvision.extension import _assert_has_ops


def roi_align(
    inp: Tensor,
    boxes: Union[Tensor, List[Tensor]],
    output_size: BroadcastingList2[int],
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> Tensor:
    # pylint: disable=too-many-arguments
    """
    Performs Region of Interest (RoI) Align operator with average pooling, as described in
    Mask R-CNN.

    Args:
        inp (Tensor[N, C, H, W]): The input tensor, i.e. a batch with ``N`` elements. Each
            element contains ``C`` feature maps of dimensions ``H x W``.
            If the tensor is quantized, we expect a batch size of ``N == 1``.
        boxes (Tensor[K, 5] or List[Tensor[L, 4]]): the box coordinates in (x1, y1, x2, y2)
            format where the regions will be taken from.
            The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
            If a single Tensor is passed, then the first column should
            contain the index of the corresponding element in the batch, i.e. a number in
            ``[0, N - 1]``.
            If a list of Tensors is passed, then each Tensor will correspond to the boxes for an
            element i in the batch.
        output_size (int or Tuple[int, int]): the size of the output (in bins or pixels) after
            the pooling is performed, as (height, width).
        spatial_scale (float): a scaling factor that maps the input coordinates to
            the box coordinates. Default: 1.0
        sampling_ratio (int): number of sampling points in the interpolation grid
            used to compute the output value of each pooled output bin. If > 0,
            then exactly ``sampling_ratio x sampling_ratio`` sampling points per bin are used. If
            <= 0, then an adaptive number of grid points are used (computed as
            ``ceil(roi_width / output_width)``, and likewise for height). Default: -1
        aligned (bool): If False, use the legacy implementation.
            If True, pixel shift the box coordinates it by -0.5 for a better alignment with the two
            neighboring pixel indices. This version is used in Detectron2

    Returns:
        Tensor[K, C, output_size[0], output_size[1]]: The pooled RoIs.
    """
    _assert_has_ops()
    check_roi_boxes_shape(boxes)
    rois = boxes
    output_size = _pair(output_size)
    if not isinstance(rois, Tensor):
        rois = convert_boxes_to_roi_format(rois)
    return torch.ops.torchvision.roi_align(
        inp,
        rois,
        spatial_scale,
        output_size[0],
        output_size[1],
        sampling_ratio,
        aligned,
    )


def _cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def convert_boxes_to_roi_format(boxes: List[Tensor]) -> Tensor:
    """Converts RoI boxes from List to Tensor format, and includes the respective box indices
    that corresponds to the element i in a batch

    Args:
        boxes (List[Tensor[L, 4]]): RoI boxes with the format of (x1, y1, x2, y2)

    Returns:
        Tensor: The converted RoI boxes
    """
    concat_boxes = _cat(boxes, dim=0)
    temp = []
    for i, box in enumerate(boxes):
        temp.append(torch.full_like(box[:, :1], i))
    ids = _cat(temp, dim=0)
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois


def check_roi_boxes_shape(boxes: Union[Tensor, List[Tensor]]) -> None:
    """Checks the RoI boxes shape, and raising error if the shape is not correct.

    If the 'boxes' is an instance of Tensor, the first column should contain the index of the
    corresponding element in the batch, i.e. a number in ``[0, N - 1]``. Hence, it should be in the
    shape Tensor[K, 5].

    Otherwise, if the 'boxes' is an instance of List, each Tensor in the list should correspond to
    the boxes for an element i in the batch. Hence, it should be in the shape List[Tensor[K, 4]],
    and the axis with dimension of 4 corresponds to the box coordinates in (x1, y1, x2, y2).

    Output type is depending on the format of proposals generated by the RPN.

    Args:
        boxes (Union[Tensor, List[Tensor]]): RoI boxes
    """
    if isinstance(boxes, (list, tuple)):
        for _tensor in boxes:
            assert (
                _tensor.size(1) == 4
            ), "The shape of the tensor in the boxes list is not correct as List[Tensor[L, 4]]"
    elif isinstance(boxes, Tensor):
        assert (
            boxes.size(1) == 5
        ), "The boxes tensor shape is not correct as Tensor[K, 5]"
    else:
        assert False, "boxes is expected to be a Tensor[L, 5] or a List[Tensor[K, 4]]"
