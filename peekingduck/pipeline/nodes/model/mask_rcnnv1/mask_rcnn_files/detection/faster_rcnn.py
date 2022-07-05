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

"""Implements Faster R-CNN model from https://arxiv.org/pdf/1506.01497.pdf.
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

from typing import Optional, Tuple, Iterable
from torch import nn, Tensor
import torch.nn.functional as F
from peekingduck.pipeline.nodes.model.mask_rcnnv1.mask_rcnn_files.ops import poolers
from peekingduck.pipeline.nodes.model.mask_rcnnv1.mask_rcnn_files.detection import (
    anchor_utils,
    rpn,
    roi_heads,
    generalized_rcnn as g_rnn,
    transform as g_rcnn_transform,
)


class FasterRCNN(g_rnn.GeneralizedRCNN):
    """
    Implements Faster R-CNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one
    for each image, and should be in 0-1 range. Different images can have different sizes.

    During inference, the model requires only the input tensors, and returns the
    post-processed predictions as a List[Dict[Tensor]], one for each input image. The fields
    of the Dict are as follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format,
          with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction

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
            trained on
        image_std (Optional[Iterable[float]]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained
            on
        rpn_anchor_generator (Optional[nn.Module]): module that generates the anchors for a set of
            feature maps.
        rpn_head (Optional[nn.Module]): module that computes the objectness and regression deltas
            from the RPN
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
    ):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )

        assert isinstance(
            rpn_anchor_generator, (anchor_utils.AnchorGenerator, type(None))
        )
        assert isinstance(box_roi_pool, (poolers.MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError(
                    "num_classes should be None when box_predictor is specified"
                )
        else:
            if box_predictor is None:
                raise ValueError(
                    "num_classes should not be None when box_predictor "
                    "is not specified"
                )

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = anchor_utils.AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = rpn.RPNHead(
                out_channels,  # type: ignore[arg-type]
                rpn_anchor_generator.num_anchors_per_location()[0],
            )

        rpn_pre_nms_top_n = dict(testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(testing=rpn_post_nms_top_n_test)

        rpn_obj = rpn.RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        if box_roi_pool is None:
            box_roi_pool = poolers.MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
            )

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,  # type: ignore[operator,arg-type]
                representation_size,
            )

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size, num_classes  # type: ignore[arg-type]
            )

        roi_heads_obj = roi_heads.RoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = g_rcnn_transform.GeneralizedRCNNTransform(
            min_size, max_size, image_mean, image_std
        )

        super().__init__(backbone, rpn_obj, roi_heads_obj, transform)


class TwoMLPHead(nn.Module):
    """Standard heads for FPN-based models. Mainly used to perform transformation on the cropped
    features from the RoI pooling function

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels: int, representation_size: int):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """Takes in input and perform forward propagation through 2 layers of fully-connected
        layers.

        Args:
            inputs (Tensor): Input Tensor

        Returns:
            Tensor: Output Tensor
        """
        inputs = inputs.flatten(start_dim=1)

        inputs = F.relu(self.fc6(inputs))
        x_out = F.relu(self.fc7(inputs))

        return x_out


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """Takes the input Tensor and pass through two separate single layer fully connected layers
            to obtain the classification logits and box regression deltas respectively.

        Args:
            inputs (Tensor): Input Tensor

        Returns:
            Tuple[Tensor, Tensor]: The classification logits and box regression deltas
        """
        if inputs.dim() == 4:
            assert list(inputs.shape[2:]) == [1, 1]
        inputs = inputs.flatten(start_dim=1)
        scores = self.cls_score(inputs)
        bbox_deltas = self.bbox_pred(inputs)

        return scores, bbox_deltas
