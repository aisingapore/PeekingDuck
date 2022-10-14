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

"""Region Proposal Network for Faster R-CNN

Modifications include:
- Removed training related code
- Removed training related arguments from RegionProposalNetwork class
    - fg_iou_thresh
    - bg_iou_thresh
    - batch_size_per_image
    - positive_fraction
- Removed tracing related code
"""

from typing import Dict, List, Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from peekingduck.pipeline.nodes.model.mask_rcnnv1.mask_rcnn_files.ops import (
    boxes as box_ops,
)
from peekingduck.pipeline.nodes.model.mask_rcnnv1.mask_rcnn_files.detection import (
    image_list,
    detection_utils as det_utils,
)


def permute_and_flatten(
    layer: Tensor, num_instance: int, channels: int, height: int, width: int
) -> Tensor:
    """Permutes a feature output to be the same format as the labels"""
    layer = layer.view(num_instance, -1, channels, height, width)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(num_instance, -1, channels)
    return layer


def concat_box_prediction_layers(
    box_cls: List[Tensor], box_regression: List[Tensor]
) -> Tuple[Tensor, Tensor]:
    """Permutes each feature level output to be the same format as labels and concatenates them on
    the first dimension"""
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        num_instance, anchors_x_channels, height, width = box_cls_per_level.shape
        anchors_x_4 = box_regression_per_level.shape[1]
        anchors = anchors_x_4 // 4
        channels = anchors_x_channels // anchors
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, num_instance, channels, height, width
        )
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, num_instance, 4, height, width
        )
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls_out = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression_out = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls_out, box_regression_out


class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for layer in self.children():
            nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
            nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]

    def forward(self, inputs: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Forward propagation for computing the objectness and bboxes' offsets from the anchor
        boxes regressions obtained from RPN

        Args:
            inputs (List[Tensor]): Input Tensor. length of inputs is the number of feature levels

        Returns:
            Tuple[List[Tensor], List[Tensor]]:
                - objectness logits. Length of list corresponds to the number of feature levels.
                    Each tensor element in the list has a shape of:
                    [
                        batch_size,
                        num_of_anchor_boxes_per_feature_level_per_location,
                        feature_height,
                        feature_width
                    ]
                - Bbox deltas regressions. Length of list corresponds to the number of feature
                    levels. Each tensor element in the list has a shape of:
                    [
                        batch_size,
                        num_of_anchor_boxes_per_feature_level_per_location x 4 corners of bbox,
                        feature_height,
                        feature_width
                    ]
        """
        logits = []
        bbox_reg = []
        for feature in inputs:
            transformed_feature = F.relu(self.conv(feature))
            logits.append(self.cls_logits(transformed_feature))
            bbox_reg.append(self.bbox_pred(transformed_feature))
        return logits, bbox_reg


class RegionProposalNetwork(nn.Module):
    """
    Implements Region Proposal Network (RPN).

    Args:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): module that computes the objectness and regression deltas
        pre_nms_top_n (Dict[str, int]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_top_n (Dict[str, int]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        score_thresh (float): Score threshold for filtering low scoring proposals
    """

    # pylint: disable=too-many-instance-attributes,too-many-arguments
    def __init__(
        self,
        anchor_generator: nn.Module,
        head: nn.Module,
        pre_nms_top_n: Dict[str, int],
        post_nms_top_n: Dict[str, int],
        nms_thresh: float,
        score_thresh: float = 0.0,
    ):
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # used during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3

    def pre_nms_top_n(self) -> int:
        """Returns the number of proposals to keep before applying NMS"""
        return self._pre_nms_top_n["testing"]

    def post_nms_top_n(self) -> int:
        """Returns the number of proposals to keep after applying NMS"""
        return self._post_nms_top_n["testing"]

    def _get_top_n_idx(
        self, objectness: Tensor, num_anchors_per_level: List[int]
    ) -> Tensor:
        """Get the top n number of proposals before applying NMS"""
        results = []
        offset = 0
        for objectness_per_level in objectness.split(num_anchors_per_level, 1):
            num_anchors = objectness_per_level.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
            _, top_n_idx = objectness_per_level.topk(pre_nms_top_n, dim=1)
            results.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(results, dim=1)

    def filter_proposals(
        self,
        proposals: Tensor,
        objectness: Tensor,
        image_shapes: List[Tuple[int, int]],
        num_anchors_per_level: List[int],
    ) -> Tuple[List[Tensor], List[Tensor]]:
        # pylint: disable=too-many-locals
        """Filters proposals through objectness thresholds, NMS, minimum size"""
        num_images = proposals.shape[0]
        device = proposals.device
        # do not backprop through objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        levels = [
            torch.full((n,), idx, dtype=torch.int64, device=device)
            for idx, n in enumerate(num_anchors_per_level)
        ]
        levels_tensor = torch.cat(levels, 0)
        levels_tensor = levels_tensor.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]

        objectness = objectness[batch_idx, top_n_idx]
        levels_tensor = levels_tensor[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(
            proposals, objectness_prob, levels, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # remove low scoring boxes
            # use >= for Backwards compatibility
            keep = torch.where(scores >= self.score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def forward(
        self,
        images: image_list.ImageList,
        features: Dict[str, Tensor],
    ) -> List[Tensor]:
        """
        Args:
            images (ImageList): images for which we want to compute the predictions
            features (OrderedDict[Tensor]): features computed from the images that are used for
                computing the predictions. Each tensor in the list correspond to different feature
                levels

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per image.
        """
        # RPN uses all feature maps that are available
        features_list = list(features.values())
        objectness, pred_bbox_deltas = self.head(features_list)
        anchors = self.anchor_generator(images, features_list)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [
            obj_per_lvl[0].shape for obj_per_lvl in objectness
        ]
        num_anchors_per_level = [
            shape[0] * shape[1] * shape[2]
            for shape in num_anchors_per_level_shape_tensors
        ]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(
            objectness, pred_bbox_deltas
        )
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, _ = self.filter_proposals(
            proposals, objectness, images.image_sizes, num_anchors_per_level
        )

        return boxes
