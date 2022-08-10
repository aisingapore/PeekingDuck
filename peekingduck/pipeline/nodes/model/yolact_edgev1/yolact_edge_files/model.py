# Copyright 2022 AI Singapore
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

# MIT License

# Copyright (c) 2020 Haotian Liu and Rafael A. Rivera-Soto

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""YolactEdge model with the feature pyramid network and prediction layer.
Modifications include:
- Yolact
    - Refactor necessary configs from yolact_edge/data/config.py
    - Removed unused conditions for ResNet and MobileNetV2 backbone
        - use_jit boolean value
    - Removed unused functions for ResNet and MobileNetV2 backbone
        - save_weights()
        - init_weights()
        - train()
        - freeze_bn()
        - fine_tune_layers()
        - extra_loss()
        - forward_flow()
        - create_embed_flow_net()
        - create_partial_backbone()
    - Removed unused functions for TRT implementation
        - _get_trt_cache_path()
        - has_trt_cached_module()
        - load_trt_cached_module()
        - save_trt_cached_module()
        - trt_load_if()
        - to_tensorrt_protonet()
        - to_tensorrt_fpn()
        - to_tensorrt_prediction_head()
        - to_tensorrt_spa()
        - to_tensorrt_flow_net()
- PredictionModule
    - Removed unused make_priors function
- Removed unused Concat class
- Removed unused PredictionModuleTRT class
- Removed unused Cat class
- Removed unused ShuffleCat class
- Removed unused ShuffleCatChunk class
- Removed unused ShuffleCatAlt class
- Removed unused FlowNetUnwrap class
- Removed unused FlowNetMiniTRTWrapper class
- Removed unused PredictionModuleTRTWrapper class
- Removed unused NoReLUBottleneck class
- Removed unused FlowNetMiniPredLayer class
- Removed unused FlowNetMiniPreConvs class
- Removed unused FlowNetMini class
- Removed unused FlowNetMiniTRT class
- Removed unused SPA class
- Removed unused FPN class
- Removed traditional NMS
"""

import logging
from math import sqrt
from itertools import product
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Union, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from peekingduck.pipeline.nodes.model.yolact_edgev1.yolact_edge_files.utils import (
    make_net,
    jaccard,
    decode,
    make_extra,
)
from peekingduck.pipeline.nodes.model.yolact_edgev1.yolact_edge_files.backbone import (
    ResNetBackbone,
    MobileNetV2Backbone,
)

if torch.cuda.is_available():
    torch.cuda.current_device()


ScriptModuleWrapper = torch.jit.ScriptModule

NUM_DOWNSAMPLE = 2
FPN_NUM_FEATURES = 256
NUM_CLASSES = 81  # For COCO classes


class YolactEdge(nn.Module):  # pylint: disable=too-many-instance-attributes
    """YolactEdge model module.

    Values for the selected backbone layers, channels, prediction aspect ratios,
    and scales are taken from the configuration file in the original repository:
    https://github.com/haotian-liu/yolact_edge/blob/master/yolact_edge/data/config.py
    """

    def __init__(
        self,
        model_type: str,
        input_size: int,
        iou_threshold: float,
        max_num_detections: int,
    ) -> None:
        super().__init__()

        self.backbone: Union[ResNetBackbone, MobileNetV2Backbone]

        if model_type[0] == "r":
            if model_type == "r101-fpn":
                self.backbone = ResNetBackbone(([3, 4, 23, 3]))
            elif model_type == "r50-fpn":
                self.backbone = ResNetBackbone(([3, 4, 6, 3]))
            self.layers = list(range(1, 4))
            src_channels = [256, 512, 1024, 2048]
        elif model_type == "mobilenetv2":
            self.backbone = MobileNetV2Backbone(
                1.0,
                [  # Based on MobileNetV2 paper's bottleneck values for t,c,n,s
                    [1, 16, 1, 1],
                    [6, 24, 2, 2],
                    [6, 32, 3, 2],
                    [6, 64, 4, 2],
                    [6, 96, 3, 1],
                    [6, 160, 3, 2],
                    [6, 320, 1, 1],
                ],
                8,
            )
            self.layers = [3, 4, 6]
            src_channels = [32, 16, 24, 32, 64, 96, 160, 320, 1280]

        num_layers = max(self.layers) + 1
        while len(self.backbone.layers) < num_layers:
            self.backbone.add_layer()

        self.num_grids = 0
        self.proto_src = 0
        mask_proto_net = (
            [(256, 3, {"padding": 1})] * 3
            + [(None, -2, {}), (256, 3, {"padding": 1})]  # type: ignore
            + [(32, 1, {})]
        )

        self.proto_net, _ = make_net(256, mask_proto_net, include_last_relu=False)

        self.fpn_phase_1 = FPNPhase1([src_channels[i] for i in self.layers])
        self.fpn_phase_2 = FPNPhase2([src_channels[i] for i in self.layers])

        self.selected_layers = list(range(len(self.layers) + NUM_DOWNSAMPLE))

        # The following values work for ResNet and MobileNetV2 backbones. Other
        # backbones may require different values.
        self.pred_aspect_ratios = [[[1, 1 / 2, 2]]] * 5
        self.pred_scales = [[24], [48], [96], [192], [384]]

        src_channels = [FPN_NUM_FEATURES] * len(self.selected_layers)
        self.prediction_layers = nn.ModuleList()

        for idx, layer_idx in enumerate(self.selected_layers):
            parent = None
            if idx > 0:
                parent = self.prediction_layers[0]
            pred = PredictionModule(
                src_channels[layer_idx],
                src_channels[layer_idx],
                aspect_ratios=self.pred_aspect_ratios[idx],
                scales=self.pred_scales[idx],  # type: ignore
                parent=parent,
                index=idx,
                input_size=input_size,
            )
            self.prediction_layers.append(pred)

        self.semantic_seg_conv = nn.Conv2d(
            src_channels[0], NUM_CLASSES - 1, kernel_size=1
        )
        self.detect = YolactEdgeHead(
            NUM_CLASSES,
            bkg_label=0,
            conf_thresh=0.05,
            iou_threshold=iou_threshold,  # This is the same as nms_thresh
            max_num_detections=max_num_detections,
        )

    def forward(self, inputs: Tensor) -> Dict[str, List]:
        """The input should be of size [batch_size, 3, img_h, img_w]

        Args:
            inputs (Tensor): The input tensor

        Returns:
            outs_wrapper (Dict): Prediction output for YolactEdge
        """
        outs_wrapper = {}
        outs = self.backbone(inputs)
        outs = [outs[i] for i in self.layers]
        outs_fpn_phase_1_wrapper = self.fpn_phase_1(*outs)
        outs_phase_1, _ = (
            outs_fpn_phase_1_wrapper[: len(outs)],
            outs_fpn_phase_1_wrapper[len(outs) :],
        )
        outs_wrapper["outs_phase_1"] = [out.detach() for out in outs_phase_1]
        outs = self.fpn_phase_2(*outs_phase_1)
        outs_wrapper["outs_phase_2"] = [out.detach() for out in outs]

        proto_x = inputs if self.proto_src is None else outs[self.proto_src]
        proto_out = self.proto_net(proto_x)
        proto_out = torch.nn.functional.relu(proto_out, inplace=True)
        proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

        pred_outs: Dict[str, Any]
        pred_outs = {"loc": [], "conf": [], "mask": [], "priors": []}

        for idx, pred_layer in zip(self.selected_layers, self.prediction_layers):
            pred_x = outs[idx]
            for key, val in pred_layer(pred_x).items():
                pred_outs[key].append(val)

        for key, val in pred_outs.items():
            pred_outs[key] = torch.cat(val, -2)

        pred_outs["proto"] = proto_out
        pred_outs["conf"] = F.softmax(pred_outs["conf"], -1)
        outs_wrapper["pred_outs"] = self.detect(pred_outs)
        return outs_wrapper

    def load_weights(self, path: Path) -> None:
        """Loads weights from a compressed save file.

        Args:
            path (Path): Path to the model weights file.

        Returns:
            YolactEdge model
        """
        state_dict = torch.load(path, map_location="cpu")
        for key in list(state_dict.keys()):
            # For backward compatibility, the new variable is called layers.
            # This has been commented out because the ResNet and MobileNetV2
            # backbones will not be using it.
            # if key.startswith("backbone.layer") and not key.startswith(
            #     "backbone.layers"
            # ):
            #     del state_dict[key]
            if key.startswith("fpn.downsample_layers."):
                if int(key.split(".")[2]) >= NUM_DOWNSAMPLE:
                    del state_dict[key]
            if key.startswith("fpn.lat_layers"):
                state_dict[key.replace("fpn.", "fpn_phase_1.")] = state_dict[key]
                del state_dict[key]
            elif key.startswith("fpn.") and key in state_dict:
                state_dict[key.replace("fpn.", "fpn_phase_2.")] = state_dict[key]
                del state_dict[key]
        self.load_state_dict(state_dict)


class PredictionModule(nn.Module):  # pylint: disable=too-many-instance-attributes
    """
    The (c) prediction module adapted from DSSD:
    https://arxiv.org/pdf/1701.06659.pdf
    Note that this is slightly different to the module in the paper
    because the Bottleneck block actually has a 3x3 convolution in
    the middle instead of a 1x1 convolution. Though, I really can't
    be arsed to implement it myself, and, who knows, this might be
    better.

    Args:
        in_channels (int): The input feature size.
        out_channels (int): The output feature size (must be a multiple of 4).
        aspect_ratios (List[List]): A list of lists of priorbox aspect ratios
            (one list per scale).
        scales (List[int]): A list of priorbox scales relative to this layer's convsize.
            For instance: If this layer has convouts of size 30x30 for an image
            of size 600x600, the 'default' (scale of 1) for this layer would
            produce bounding boxes with an area of 20x20px. If the scale is .5
            on the other hand, this layer would consider bounding boxes with area
            10x10px, etc.
        parent (PredictionModule): If parent is a PredictionModule, this module
            will use all the layers from parent instead of from this module.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_channels: int,
        out_channels: int = 1024,
        aspect_ratios: Iterable[Any] = None,
        scales: Iterable[List[int]] = None,
        parent: Optional[Callable] = None,  # PredictionModule
        index: int = 0,
        input_size: int = None,
    ) -> None:

        super().__init__()
        self.params = [in_channels, out_channels, aspect_ratios, scales, parent, index]
        self.num_classes = NUM_CLASSES
        self.mask_dim = 32
        self.num_priors = sum(len(x) for x in aspect_ratios)  # type: ignore
        self.parent = [parent]
        self.index = index
        head_layer_params: Dict[Any, Any] = {"kernel_size": 3, "padding": 1}

        if parent is None:
            self.upfeature, out_channels = make_net(
                in_channels, [(256, 3, {"padding": 1})]
            )
            self.bbox_layer = nn.Conv2d(
                out_channels, self.num_priors * 4, **head_layer_params
            )
            self.conf_layer = nn.Conv2d(
                out_channels, self.num_priors * self.num_classes, **head_layer_params
            )
            self.mask_layer = nn.Conv2d(
                out_channels, self.num_priors * self.mask_dim, **head_layer_params
            )
            self.bbox_extra, self.conf_extra, self.mask_extra = [
                make_extra(x, out_channels) for x in (0, 0, 0)
            ]

        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.priors: Union[None, Tensor] = None
        self.last_conv_size = (0, 0)
        self.input_size = input_size

    def forward(self, inputs: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            inputs (Tensor): The convout from a layer in the backbone network of
                size: [batch_size, in_channels, conv_h, conv_w])

        Returns a dictionary of Tensors with the following sizes:
            bbox_coords (Tensor): [batch_size, conv_h*conv_w*num_priors, 4]
            class_confs (Tensor): [batch_size, conv_h*conv_w*num_priors, num_classes]
            mask_output (Tensor): [batch_size, conv_h*conv_w*num_priors, mask_dim]
            prior_boxes (Tensor): [conv_h*conv_w*num_priors, 4]
        """
        src = self if self.parent[0] is None else self.parent[0]
        conv_h = inputs.size(2)
        conv_w = inputs.size(3)

        inputs = src.upfeature(inputs)

        bbox_x = src.bbox_extra(inputs)
        conf_x = src.conf_extra(inputs)
        mask_x = src.mask_extra(inputs)

        bbox = (
            src.bbox_layer(bbox_x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(inputs.size(0), -1, 4)
        )
        conf = (
            src.conf_layer(conf_x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(inputs.size(0), -1, self.num_classes)
        )
        mask = (
            src.mask_layer(mask_x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(inputs.size(0), -1, self.mask_dim)
        )

        mask = torch.tanh(mask)

        priors = self.make_priors(conv_h, conv_w)
        preds = {"loc": bbox, "conf": conf, "mask": mask, "priors": priors}
        return preds

    def make_priors(self, conv_h: int, conv_w: int) -> Optional[Tensor]:
        """Priors are [center-x, center-y, width, height] where center-x and
        center-y are the center coordinates of the box.

        Args:
            conv_h (int): The height of the convolutional output.
            conv_w (int): The width of the convolutional output.

        Returns:
            self.priors (Tensor): [conv_h * conv_w * num_priors, 4]
        """
        if self.last_conv_size != (conv_w, conv_h):
            prior_data = []
            for j, i in product(range(conv_h), range(conv_w)):
                # +0.5 because priors are in center-size notation
                center_x = (i + 0.5) / conv_w
                center_y = (j + 0.5) / conv_h
                for scale, aspect_ratios in zip(self.scales, self.aspect_ratios):  # type: ignore
                    for aspect_ratio in aspect_ratios:
                        aspect_ratio = sqrt(aspect_ratio)
                        width = scale * aspect_ratio / self.input_size
                        height = width
                        prior_data += [center_x, center_y, width, height]
            self.priors = Tensor(prior_data).view(-1, 4)
            self.last_conv_size = (conv_w, conv_h)
        return self.priors


class FPNPhase1(ScriptModuleWrapper):
    """First phase of the feature pyramid network"""

    __constants__ = ["interpolation_mode"]

    def __init__(self, in_channels: List[int]) -> None:
        super().__init__()
        self.src_channels = in_channels
        self.lat_layers = nn.ModuleList(
            [
                nn.Conv2d(x, FPN_NUM_FEATURES, kernel_size=1)
                for x in reversed(in_channels)
            ]
        )
        self.interpolation_mode = "bilinear"

    def forward(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        x_1: Optional[torch.Tensor] = None,
        x_2: Optional[torch.Tensor] = None,
        x_3: Optional[torch.Tensor] = None,
        x_4: Optional[torch.Tensor] = None,
        x_5: Optional[torch.Tensor] = None,
        x_6: Optional[torch.Tensor] = None,
        x_7: Optional[torch.Tensor] = None,
    ) -> List[Tensor]:
        """
        Args:
            - convouts (List): A list of convouts for the corresponding layers
                in in_channels.
        Returns:
            - out (List(Tensor)): A list of FPN convouts in the same order as x
                with extra downsample layers if requested.
        """
        convouts_ = [x_1, x_2, x_3, x_4, x_5, x_6, x_7]
        convouts = []

        for count, _ in enumerate(convouts_):
            if convouts_[count] is not None:
                convouts.append(convouts_[count])

        out = []
        lat_feats = []
        x_0 = torch.zeros(1)

        for i in range(len(convouts)):
            out.append(x_0)
            lat_feats.append(x_0)

        count = len(convouts)
        for lat_layer in self.lat_layers:
            count -= 1
            if count < len(convouts) - 1:
                _, _, height, weight = convouts[count].size()  # type: ignore
                x_0 = F.interpolate(
                    x_0,
                    size=(height, weight),
                    mode=self.interpolation_mode,
                    align_corners=False,
                )
            lat_iter = lat_layer(convouts[count])
            lat_feats[count] = lat_iter

            x_0 = x_0 + lat_iter

            out[count] = x_0

        for i in range(len(convouts)):
            out.append(lat_feats[i])
        return out


class FPNPhase2(ScriptModuleWrapper):
    """Second phase of the feature pyramid network"""

    __constants__ = ["num_downsample"]

    def __init__(self, in_channels: List[int]) -> None:
        super().__init__()
        self.src_channels = in_channels
        self.pred_layers = nn.ModuleList(
            [
                nn.Conv2d(FPN_NUM_FEATURES, FPN_NUM_FEATURES, kernel_size=3, padding=1)
                for _ in in_channels
            ]
        )
        self.num_downsample = NUM_DOWNSAMPLE
        self.downsample_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    FPN_NUM_FEATURES,
                    FPN_NUM_FEATURES,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                )
                for _ in range(self.num_downsample)
            ]
        )

    def forward(  # pylint: disable=too-many-arguments
        self,
        x_1: Optional[List] = None,
        x_2: Optional[List] = None,
        x_3: Optional[List] = None,
        x_4: Optional[List] = None,
        x_5: Optional[List] = None,
        x_6: Optional[List] = None,
        x_7: Optional[List] = None,
    ) -> List[Optional[List[Any]]]:
        """
        Args:
            x_1, x_2, x_3, x_4, x_5, x_6 (List): A list of convouts for the
                corresponding layers in in_channels.

        Returns:
            out (List): A list of FPN convouts in the same order as x with extra
                downsample layers if requested.
        """
        out_ = [x_1, x_2, x_3, x_4, x_5, x_6, x_7]
        out = []
        for count, _ in enumerate(out_):
            if out_[count] is not None:
                out.append(out_[count])

        j = len(out)
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = F.relu(pred_layer(out[j]))  # type: ignore
        for downsample_layer in self.downsample_layers:
            out.append(downsample_layer(out[-1]))
        return out


class YolactEdgeHead:
    """
    This is the final layer of Single Shot Detection (SSD). Decode location preds,
    apply non-maximum suppression to location predictions based on conf scores and
    threshold to a top maximum number output predictions for both confidence scores
    and locations, as the predicted masks.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        num_classes: int,
        bkg_label: int,
        conf_thresh: float,
        iou_threshold: float,
        max_num_detections: int,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.iou_threshold = iou_threshold
        self.conf_thresh = conf_thresh
        self.max_num_detections = max_num_detections

    def __call__(self, predictions: Dict[str, torch.Tensor]) -> List[Any]:
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
            mask_data: (tensor) Mask preds from mask layers
                Shape: [batch, num_priors, mask_dim]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors, 4]
            proto_data: (tensor) If using mask_type.lincomb, the prototype masks
                Shape: [batch, mask_h, mask_w, mask_dim]

        Returns:
            out (List): output of shape (batch_size, max_num_detections, 1 + 1 + 4 + mask_dim)
                These outputs are in the order: class idx, confidence, bbox coords,
                and mask.
        """
        loc_data = predictions["loc"]
        conf_data = predictions["conf"]
        mask_data = predictions["mask"]
        prior_data = predictions["priors"]
        proto_data = predictions["proto"] if "proto" in predictions else None

        out = []
        batch_size = loc_data.size(0)
        num_priors = prior_data.size(0)
        conf_preds = (
            conf_data.view(batch_size, num_priors, self.num_classes)
            .transpose(2, 1)
            .contiguous()
        )

        for batch_idx in range(batch_size):
            decoded_boxes = decode(loc_data[batch_idx], prior_data)
            result = self.detect(batch_idx, conf_preds, decoded_boxes, mask_data)
            if result is not None and proto_data is not None:
                result["proto"] = proto_data[batch_idx]
            out.append(result)
        return out

    def detect(  # pylint: disable=too-many-arguments
        self,
        batch_idx: int,
        conf_preds: Tensor,
        decoded_boxes: Tensor,
        mask_data: Tensor,
    ) -> Optional[Dict[str, Any]]:
        """Perform nms for only the max scoring class that isn't background (class 0)

        Args:
            batch_idx: (int) The batch index
            conf_preds: (Tensor) Confidence predictions for each prior
            decoded_boxes: (Tensor) Decoded boxes for each prior
            mask_data: (Tensor) Mask predictions for each prior

        Returns:
            A dictionary containing the following keys:
                box (Tensor): The bounding box values for each detection (0 to 1)
                mask (Tensor): The segmentation mask for each detection (-1 to 1)
                class (Tensor): Class ID for each detection (0 to 80)
                score (Tensor): Confidence score for each detection (0 to 1)
        """
        cur_scores = conf_preds[batch_idx, 1:, :]
        conf_scores, _ = torch.max(cur_scores, dim=0)
        keep = conf_scores > self.conf_thresh
        scores = cur_scores[:, keep]
        boxes = decoded_boxes[keep, :]
        masks = mask_data[batch_idx, keep, :]

        if scores.size(1) == 0:
            return None

        boxes, masks, classes, scores = self.fast_nms(
            boxes,
            masks,
            scores,
            self.iou_threshold,
            self.max_num_detections,
        )

        return {"box": boxes, "mask": masks, "class": classes, "score": scores}

    @classmethod
    def fast_nms(  # pylint: disable=too-many-arguments
        cls,
        boxes: Tensor,
        masks: Tensor,
        scores: Tensor,
        iou_threshold: float,
        max_num_detections: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Non-maximum Suppression

        Args:
            boxes (Tensor): Bounding boxes for each object
            masks (Tensor): Masks for each object
            scores (Tensor): Confidence scores for each object
            iou_threshold (float): IoU threshold for NMS
            max_num_detections (int): Maximum number of detections

        Returns:
            boxes (np.ndarray): array of detected bboxes
            masks (np.ndarray): array of detected masks
            classes (np.ndarray): array of class labels
            scores (np.ndarray): array of detection confidence scores
        """
        scores, idx = scores.sort(1, descending=True)
        idx = idx[:, :max_num_detections].contiguous()
        scores = scores[:, :max_num_detections]
        num_classes, num_dets = idx.size()
        boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
        masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)

        iou = jaccard(boxes, boxes)
        iou.triu_(diagonal=1)
        iou_max, _ = iou.max(dim=1)

        # Filter out the ones that are higher that the IoU threshold
        keep = iou_max <= iou_threshold

        classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(
            keep
        )
        classes = classes[keep]

        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]

        scores, idx = scores.sort(0, descending=True)
        idx = idx[:max_num_detections]
        scores = scores[:max_num_detections]
        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]

        return boxes, masks, classes, scores
