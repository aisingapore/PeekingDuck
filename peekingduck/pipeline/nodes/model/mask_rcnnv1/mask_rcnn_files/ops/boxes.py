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

"""Box operations for Region Proposal Network
Modifications include:
- Removed unused functions from original script
    - _box_inter_union
    - box_iou
- Removed tracing related code
"""

from typing import Tuple
import torch
from torch import Tensor
import torchvision
from torchvision.extension import _assert_has_ops


# pylint: disable=protected-access
def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than iou_threshold with another (higher scoring)
    box.

    If multiple boxes have the exact same score and satisfy the IoU
    criterion with respect to a reference box, the selected box is
    not guaranteed to be the same between CPU and GPU. This is similar
    to the behavior of argsort in PyTorch when repeated values are present.

    Args:
        boxes (Tensor[N, 4])): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    _assert_has_ops()
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def clip_boxes_to_image(boxes: Tensor, size: Tuple[int, int]) -> Tensor:
    """
    Clip boxes so that they lie inside an image of size `size`.

    Args:
        boxes (Tensor[N, 4]): boxes in ``(x1, y1, x2, y2)`` format
            with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        size (Tuple[height, width]): size of the image

    Returns:
        Tensor[N, 4]: clipped boxes
    """
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    height, width = size

    boxes_x = boxes_x.clamp(min=0, max=width)
    boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def remove_small_boxes(boxes: Tensor, min_size: float) -> Tensor:
    """
    Remove boxes which contains at least one side smaller than min_size.

    Args:
        boxes (Tensor[N, 4]): boxes in ``(x1, y1, x2, y2)`` format
            with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        min_size (float): minimum size

    Returns:
        Tensor[K]: indices of the boxes that have both sides
        larger than min_size
    """
    widths, heights = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = (widths >= min_size) & (heights >= min_size)
    keep = torch.where(keep)[0]
    return keep


def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 4]): boxes where NMS will be performed. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        idxs (Tensor[N]): indices of the categories for each one of the boxes.
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    # Benchmarks that drove the following thresholds are at
    # https://github.com/pytorch/vision/issues/1311#issuecomment-781329339
    # Ideally for GPU we'd use a higher threshold
    if boxes.numel() > 4000 and not torchvision._is_tracing():
        return _batched_nms_vanilla(boxes, scores, idxs, iou_threshold)

    return _batched_nms_coordinate_trick(boxes, scores, idxs, iou_threshold)


@torch.jit._script_if_tracing
def _batched_nms_coordinate_trick(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    """strategy: in order to perform NMS independently per class,
    we add an offset to all the boxes. The offset is dependent
    only on the class idx, and is large enough so that boxes
    from different classes do not overlap"""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


@torch.jit._script_if_tracing
def _batched_nms_vanilla(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    # Based on Detectron2 implementation, just manually call nms() on each class independently
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    for class_id in torch.unique(idxs):
        curr_indices = torch.where(idxs == class_id)[0]
        curr_keep_indices = nms(
            boxes[curr_indices], scores[curr_indices], iou_threshold
        )
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = torch.where(keep_mask)[0]
    return keep_indices[scores[keep_indices].sort(descending=True)[1]]


def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        Tensor[N]: the area for each box
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _upcast(input_tensor: Tensor) -> Tensor:
    """Protects from numerical overflows in multiplications by upcasting to the equivalent
    higher type"""
    if input_tensor.is_floating_point():
        return (
            input_tensor
            if input_tensor.dtype in (torch.float32, torch.float64)
            else input_tensor.float()
        )

    return (
        input_tensor
        if input_tensor.dtype in (torch.int32, torch.int64)
        else input_tensor.int()
    )
