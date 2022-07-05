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

"""Detection Utilities for Region Proposal Network and Region of Interest Operations
Modifications include:
- Removed unused methods from BoxCoder class
    - encode
    - encode_single
    - encode_boxes
"""

from typing import List, Tuple
import math
import torch
from torch import Tensor


class BoxCoder:
    """
    This class decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(
        self,
        weights: Tuple[float, float, float, float],
        bbox_xform_clip: float = math.log(1000.0 / 16),
    ) -> None:
        """
        Args:
            weights (4-element tuple): Each element corresponds to the weights that the raw
                regressed offsets (dx, dy, dw, dh) will be divided by. (i.e. the higher the weight,
                the less effect the regressed offsets have on the boxes)
            bbox_xform_clip (float): The log of maximum allowable width or height scale offset.
                E.g. if maximum allowable width and height offset is 3 times of original width or
                height, the bbox_xform_clip will be log(3). It is also to prevent sending too large
                an input to torch.exp() when calculating the predicted width and height.
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def decode(self, rel_codes: Tensor, boxes: List[Tensor]) -> Tensor:
        """
        From a list of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Args:
            rel_codes (Tensor): encoded boxes offsets regressions
            boxes (List[Tensor]): List of reference boxes.

        Returns:
            Tensor: Decoded boxes
        """
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, Tensor)
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        if box_sum > 0:
            rel_codes = rel_codes.reshape(box_sum, -1)
        pred_boxes = self.decode_single(rel_codes, concat_boxes)
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)
        return pred_boxes

    def decode_single(self, rel_codes: Tensor, boxes: Tensor) -> Tensor:
        # pylint: disable=too-many-locals
        """Performs decoding operation for the boxes

        Args:
            rel_codes (Tensor): encoded boxes offsets regressions
            boxes (Tensor): Reference boxes

        Returns:
            Tensor: Decoded boxes
        """
        boxes = boxes.to(rel_codes.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        weight_x, weight_y, weight_w, weight_h = self.weights
        del_x = rel_codes[:, 0::4] / weight_x
        del_y = rel_codes[:, 1::4] / weight_y
        del_w = rel_codes[:, 2::4] / weight_w
        del_h = rel_codes[:, 3::4] / weight_h

        # Prevent sending too large values into torch.exp()
        del_w = torch.clamp(del_w, max=self.bbox_xform_clip)
        del_h = torch.clamp(del_h, max=self.bbox_xform_clip)

        pred_ctr_x = del_x * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = del_y * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(del_w) * widths[:, None]
        pred_h = torch.exp(del_h) * heights[:, None]

        # Distance from center to box's corner.
        c_to_c_h = (
            torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        )
        c_to_c_w = (
            torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        )

        pred_boxes1 = pred_ctr_x - c_to_c_w
        pred_boxes2 = pred_ctr_y - c_to_c_h
        pred_boxes3 = pred_ctr_x + c_to_c_w
        pred_boxes4 = pred_ctr_y + c_to_c_h
        pred_boxes = torch.stack(
            (pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2
        ).flatten(1)
        return pred_boxes
