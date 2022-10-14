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

"""Region of Interest Heads for Faster R-CNN.
Modifications include:
- Removed training, target and losses related code
- Removed keypoint detection related codes and arguments.
- Removed ONNX related code
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from peekingduck.pipeline.nodes.model.mask_rcnnv1.mask_rcnn_files.detection import (
    detection_utils as det_utils,
)
from peekingduck.pipeline.nodes.model.mask_rcnnv1.mask_rcnn_files.ops import (
    boxes as box_ops,
)


def maskrcnn_inference(mask_logits: Tensor, labels: List[Tensor]) -> List[Tensor]:
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    Args:
        mask_logits (Tensor): The mask logits
        labels (list[BoxList]): bounding boxes that are used as
            reference, one for each image

    Returns:
        results (list[BoxList]): one BoxList for each image, containing
            the extra field mask
    """
    mask_prob = mask_logits.sigmoid()

    # select masks corresponding to the predicted classes
    num_masks = mask_logits.shape[0]
    boxes_per_image = [label.shape[0] for label in labels]
    labels_ = torch.cat(labels)
    index = torch.arange(num_masks, device=labels_.device)
    mask_prob = mask_prob[index, labels_][:, None]
    mask_prob_out = mask_prob.split(boxes_per_image, dim=0)

    return mask_prob_out


def expand_boxes(boxes: Tensor, scale: float) -> Tensor:
    """Enlarge boxes with the specified scale"""
    w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
    h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
    x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
    y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


def expand_masks(mask: Tensor, padding: int) -> Tuple[Tensor, float]:
    """Enlarge masks with the specified padding and returns the resultant scale"""
    mask_width = mask.shape[-1]
    scale = float(mask_width + 2 * padding) / mask_width
    padded_mask = F.pad(mask, (padding,) * 4)
    return padded_mask, scale


def paste_mask_in_image(mask: Tensor, box: Tensor, im_h: int, im_w: int) -> Tensor:
    """Resize mask to same size as the bounding box and paste onto an image-size frame that is
    initialized to zero. The location of the mask on the frame follows the location of the bounding
    box"""
    to_remove = 1
    width = int(box[2] - box[0] + to_remove)
    height = int(box[3] - box[1] + to_remove)
    width = max(width, 1)
    height = max(height, 1)

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))

    # Resize mask
    mask = F.interpolate(
        mask, size=(height, width), mode="bilinear", align_corners=False
    )
    mask = mask[0][0]

    im_mask = torch.zeros((im_h, im_w), dtype=mask.dtype, device=mask.device)
    x_0 = max(box[0], 0)  # type: ignore[call-overload]
    x_1 = min(box[2] + 1, im_w)  # type: ignore[call-overload]
    y_0 = max(box[1], 0)  # type: ignore[call-overload]
    y_1 = min(box[3] + 1, im_h)  # type: ignore[call-overload]

    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])
    ]
    return im_mask


def paste_masks_in_image(
    masks: Tensor, boxes: Tensor, img_shape: Tuple[int, int], padding: int = 1
) -> Tensor:
    """Expands masks and bounding boxes based on the padding size and paste masks onto a
    zero-initialized frame and a location based on the bounding boxes using the
    `paste_mask_in_image()` function"""
    masks, scale = expand_masks(masks, padding=padding)
    boxes = expand_boxes(boxes, scale).to(dtype=torch.int64)
    im_h, im_w = img_shape

    res = [paste_mask_in_image(m[0], b, im_h, im_w) for m, b in zip(masks, boxes)]
    if len(res) > 0:
        ret = torch.stack(res, dim=0)[:, None]
    else:
        ret = masks.new_empty((0, 1, im_h, im_w))
    return ret


class RoIHeads(nn.Module):
    """A class for Region of Interest Head for Mask-RCNN

    Args:
        box_roi_pool (nn.Module): (MultiScaleRoIAlign): the module which crops and resizes the
            feature maps in the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the
            encoding/decoding of the bounding boxes
        score_thresh (float): during inference, only return proposals with a classification
            score greater than box_score_thresh
        nms_thresh (float): NMS threshold for the prediction head.
        detections_per_img (int): maximum number of detections per image, for all classes.
        mask_roi_pool (nn.Module, optional): the module which crops and resizes the
            feature maps in the locations indicated by the bounding boxes, which will be used for
            the mask head. Defaults to None.
        mask_head (nn.Module, optional): module that takes the cropped feature maps as
            input. Defaults to None.
        mask_predictor (nn.Module, optional): module that takes the output of the
            mask_head and returns the segmentation mask logits. Defaults to None.
    """

    # pylint: disable=too-many-instance-attributes,too-many-arguments
    def __init__(
        self,
        box_roi_pool: nn.Module,
        box_head: nn.Module,
        box_predictor: nn.Module,
        bbox_reg_weights: Union[Tuple[float, float, float, float], None],
        # Faster R-CNN inference
        score_thresh: float,
        nms_thresh: float,
        detections_per_img: int,
        # Mask
        mask_roi_pool: Optional[nn.Module] = None,
        mask_head: Optional[nn.Module] = None,
        mask_predictor: Optional[nn.Module] = None,
    ):
        super().__init__()

        if bbox_reg_weights is None:
            bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor

    def has_mask(self) -> bool:
        """Checks whether the RoI heads have mask_roi_pool, mask_head and mask_predictor

        Returns:
            bool: True if the RoI heads have mask_roi_pool, mask_head and mask_predictor
        """
        if self.mask_roi_pool is None:
            return False
        if self.mask_head is None:
            return False
        if self.mask_predictor is None:
            return False
        return True

    def postprocess_detections(
        self,
        class_logits: Tensor,
        box_regression: Tensor,
        proposals: List[Tensor],
        image_shapes: List[Tuple[int, int]],
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        # pylint: disable=too-many-locals
        """Perform postprocessing for detection results"""
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(
            pred_boxes_list, pred_scores_list, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(
        self,
        features: Dict[str, Tensor],
        proposals: List[Tensor],
        image_shapes: List[Tuple[int, int]],
    ) -> List[Dict[str, Tensor]]:
        # pylint: disable=too-many-locals
        """Forward propagation for the RoIHead.
        Takes the feature and proposals, and predicts the bounding boxes, labels, scores and
        predicts the masks if there are any mask_roi_pool, mask_head and mask_predictor present.

        Args:
            features (Dict[str, Tensor]): Features from the backbone
            proposals (List[Tensor]): Proposals from the RPN
            image_shapes (List[Tuple[int, int]]): Original sizes of the input images

        Raises:
            Exception: If mask_roi_pool is not present even after has_mask() return True

        Returns:
            List[Dict[str, Tensor]]: A list of dictionary of prediction outputs. The keys in the
                dictionary are:
                - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with
                    0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                - labels (Int64Tensor[N]): the predicted labels for each image
                - scores (Tensor[N]): the scores of each prediction
                - masks (FloatTensor[N, 1, H, W]): the predicted masks for each instance with
                    probability values [0, 1]
        """
        labels = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        results: List[Dict[str, torch.Tensor]] = []
        boxes, scores, labels = self.postprocess_detections(
            class_logits, box_regression, proposals, image_shapes
        )
        num_images = len(boxes)
        for i in range(num_images):
            results.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )

        if self.has_mask():
            mask_proposals = [result["boxes"] for result in results]

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(
                    features, mask_proposals, image_shapes
                )
                mask_features = self.mask_head(mask_features)  # type: ignore[misc]
                mask_logits = self.mask_predictor(mask_features)  # type: ignore[misc]
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            labels = [result["labels"] for result in results]
            masks_probs = maskrcnn_inference(mask_logits, labels)
            for mask_prob, result in zip(masks_probs, results):
                result["masks"] = mask_prob

        return results
