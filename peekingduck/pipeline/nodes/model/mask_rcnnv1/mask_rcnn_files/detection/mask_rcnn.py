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

"""Implements the Mask R-CNN model.
Modifications include:
- Removed training related arguments for ROI_Heads and __init__:
    - box_fg_iou_thresh
    - box_bg_iou_thresh
    - box_batch_size_per_image
    - box_positive_fraction
- Removed training related arguments for RegionProposalNetwork and __init__:
    - fg_iou_thresh
    - bg_iou_thresh
    - batch_size_per_image
    - positive_fraction
    - rpn_pre_nms_top_n_train
    - rpn_post_nms_top_n_train
"""

from typing import Iterable, Optional, Tuple
from collections import OrderedDict
from torch import nn
from peekingduck.pipeline.nodes.model.mask_rcnnv1.mask_rcnn_files.ops import poolers
from peekingduck.pipeline.nodes.model.mask_rcnnv1.mask_rcnn_files.detection import (
    faster_rcnn,
)


class MaskRCNN(faster_rcnn.FasterRCNN):
    """
    Implements Mask R-CNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for
    each image, and should be in 0-1 range. Different images can have different sizes.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores of each prediction
        - masks (UInt8Tensor[N, 1, H, W]): the predicted masks for each instance, in 0-1 range.
          In order to obtain the final segmentation masks, the soft masks can be thresholded,
          generally with a value of 0.5 (mask >= 0.5)

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Optional[Iterable[float]]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been
            trained on. Requires 3 floating point elements in the Iterable
        image_std (Optional[Iterable[float]]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained
            on. Requires 3 floating point elements in the Iterable
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of
            feature maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the
            RPN
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during
            testing
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during
            testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_score_thresh (float): during inference, only return proposals with a classification
            score greater than rpn_score_thresh
        box_roi_pool (Optional[nn.Module]): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (Optional[nn.Module]): module that takes the cropped feature maps as input
        box_predictor (Optional[nn.Module]): module that takes the output of box_head and returns
            the classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification
            score greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        bbox_reg_weights (Optional[Tuple[float, float, float, float]]): weights for the
            encoding/decoding of the bounding boxes
        mask_roi_pool (Optional[nn.Module]): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes, which will be used for the mask head.
        mask_head (Optional[nn.Module]): module that takes the cropped feature maps as input
        mask_predictor (Optional[nn.Module]): module that takes the output of the mask_head and
            returns the segmentation mask logits
    """

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: Optional[int] = None,
        # transform parameters
        min_size: int = 800,
        max_size: int = 1333,
        image_mean: Optional[Iterable[float]] = None,
        image_std: Optional[Iterable[float]] = None,
        # RPN parameters
        rpn_anchor_generator: Optional[nn.Module] = None,
        rpn_head: Optional[nn.Module] = None,
        rpn_pre_nms_top_n_test: int = 1000,
        rpn_post_nms_top_n_test: int = 1000,
        rpn_nms_thresh: float = 0.7,
        rpn_score_thresh: float = 0.0,
        # Box parameters
        box_roi_pool: Optional[nn.Module] = None,
        box_head: Optional[nn.Module] = None,
        box_predictor: Optional[nn.Module] = None,
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        box_detections_per_img: int = 100,
        bbox_reg_weights: Optional[Tuple[float, float, float, float]] = None,
        # Mask parameters
        mask_roi_pool: Optional[nn.Module] = None,
        mask_head: Optional[nn.Module] = None,
        mask_predictor: Optional[nn.Module] = None,
    ):

        assert isinstance(mask_roi_pool, (poolers.MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if mask_predictor is not None:
                raise ValueError(
                    "num_classes should be None when mask_predictor is specified"
                )

        out_channels = backbone.out_channels

        if mask_roi_pool is None:
            mask_roi_pool = poolers.MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2
            )

        if mask_head is None:
            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            mask_head = MaskRCNNHeads(
                out_channels, mask_layers, mask_dilation  # type: ignore[arg-type]
            )

        if mask_predictor is None:
            mask_predictor_in_channels = 256  # == mask_layers[-1]
            mask_dim_reduced = 256
            mask_predictor = MaskRCNNPredictor(
                mask_predictor_in_channels, mask_dim_reduced, num_classes  # type: ignore[arg-type]
            )

        super().__init__(
            backbone,
            num_classes,
            # transform parameters
            min_size,
            max_size,
            image_mean,
            image_std,
            # RPN-specific parameters
            rpn_anchor_generator,
            rpn_head,
            rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_score_thresh,
            # Box parameters
            box_roi_pool,
            box_head,
            box_predictor,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            bbox_reg_weights,
        )

        self.roi_heads.mask_roi_pool = mask_roi_pool
        self.roi_heads.mask_head = mask_head
        self.roi_heads.mask_predictor = mask_predictor


class MaskRCNNHeads(nn.Sequential):
    """Implements the head for Mask R-CNN, a module that takes the cropped feature maps as input"""

    def __init__(self, in_channels: int, layers: Iterable[int], dilation: int):
        """
        Args:
            in_channels (int): number of input channels
            layers (Iterable[int]): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
        """
        layers_dict: "OrderedDict[str, nn.Module]" = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            layers_dict["mask_fcn{}".format(layer_idx)] = nn.Conv2d(
                next_feature,
                layer_features,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
            )
            layers_dict["relu{}".format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features

        super().__init__(layers_dict)
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")


class MaskRCNNPredictor(nn.Sequential):
    """A module that takes the output of the mask_head and returns the
    segmentation mask logits"""

    def __init__(self, in_channels: int, dim_reduced: int, num_classes: int):
        super().__init__(
            OrderedDict(
                [
                    (
                        "conv5_mask",
                        nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0),
                    ),
                    ("relu", nn.ReLU(inplace=True)),
                    ("mask_fcn_logits", nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
                ]
            )
        )

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
