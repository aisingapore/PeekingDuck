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
    - Refactor configs
    - Removed unused conditions for ResNet and MobileNetV2 backbone
        - use_jit boolean value
        - 
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
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import product
from math import sqrt
from typing import Dict, List, Tuple, Any
from torch import Tensor

from peekingduck.pipeline.nodes.model.yolact_edgev1.yolact_edge_files.utils import (
    make_net, 
    SELECTED_LAYERS,
    SRC_CHANNELS,
    NUM_DOWNSAMPLE,
    FPN_NUM_FEATURES,
    NUM_CLASSES,
    MAX_NUM_DETECTIONS)
from peekingduck.pipeline.nodes.model.yolact_edgev1.yolact_edge_files.backbone import (
    construct_backbone)

try:
    torch.cuda.current_device()
except:
    pass

ScriptModuleWrapper = torch.jit.ScriptModule

class YolactEdge(nn.Module):
    """YolactEdge model module.
    """
    def __init__(self) -> None:
        super().__init__()
        self.backbone = construct_backbone()
        self.num_grids = 0
        self.proto_src = 0
        src_channels = SRC_CHANNELS
        mask_proto_net = [(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), 
                          (256, 3, {'padding': 1})] + [(32, 1, {})]
        
        self.proto_net, _ = make_net(256, mask_proto_net, include_last_relu=False)
        self.selected_layers = SELECTED_LAYERS

        # src_channels = [32, 16, 24, 32, 64, 96, 160, 320, 1280] # MobileNetV2
        self.fpn_phase_1 = FPN_phase_1([src_channels[i] for i in self.selected_layers])
        self.fpn_phase_2 = FPN_phase_2([src_channels[i] for i in self.selected_layers])
        
        self.selected_layers = list(
            range(len(self.selected_layers) + NUM_DOWNSAMPLE)
        ) # R50 / R101

        self.pred_aspect_ratios = [[[1, 1/2, 2]]]*5
        self.pred_scales = [[24], [48], [96], [192], [384]]
        src_channels = [FPN_NUM_FEATURES] * len(self.selected_layers)
        
        self.prediction_layers = nn.ModuleList()

        for idx, layer_idx in enumerate(self.selected_layers):
            parent = None
            if idx > 0:
                parent = self.prediction_layers[0]
            pred = PredictionModule(src_channels[layer_idx], src_channels[layer_idx],
                                    aspect_ratios = self.pred_aspect_ratios[idx],
                                    scales        = self.pred_scales[idx],
                                    parent        = parent,
                                    index         = idx)
            self.prediction_layers.append(pred)

        self.semantic_seg_conv = nn.Conv2d(
            src_channels[0], NUM_CLASSES-1, kernel_size=1
        )

        self.detect = Detector(
            NUM_CLASSES, bkg_label=0, top_k=200, conf_thresh=0.05, nms_thresh=0.5
        )

    def forward(self, x, extras=None):
        """ The input should be of size [batch_size, 3, img_h, img_w] """
        outs_wrapper = {}
        outs = self.backbone(x)
        outs = [outs[i] for i in SELECTED_LAYERS]
        outs_fpn_phase_1_wrapper = self.fpn_phase_1(*outs)
        outs_phase_1, _ = outs_fpn_phase_1_wrapper[
            :len(outs)], outs_fpn_phase_1_wrapper[len(outs):]
        outs_wrapper["outs_phase_1"] = [out.detach() for out in outs_phase_1]
        outs = self.fpn_phase_2(*outs_phase_1)
        outs_wrapper["outs_phase_2"] = [out.detach() for out in outs]

        proto_out = None
        proto_x = x if self.proto_src is None else outs[self.proto_src]
        proto_out = self.proto_net(proto_x)
        proto_out = torch.nn.functional.relu(proto_out, inplace=True)
        proto_out = proto_out.permute(0, 2, 3, 1).contiguous()
        pred_outs = {'loc': [], 'conf': [], 'mask': [], 'priors': []}
        
        for idx, pred_layer in zip(self.selected_layers, self.prediction_layers):
            pred_x = outs[idx]
            p = pred_layer(pred_x)
            for k, v in p.items():
                pred_outs[k].append(v)
                    
        for k, v in pred_outs.items():
            pred_outs[k] = torch.cat(v, -2)

        pred_outs['proto'] = proto_out
        pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)
        outs_wrapper["pred_outs"] = self.detect(pred_outs)

        return outs_wrapper

    def load_weights(self, path):
        state_dict = torch.load(path, map_location='cpu')
        for key in list(state_dict.keys()):
            if key.startswith('backbone.layer') and not key.startswith(
                'backbone.layers'):
                del state_dict[key]
            if key.startswith('fpn.downsample_layers.'):
                if int(key.split('.')[2]) >= NUM_DOWNSAMPLE:
                    del state_dict[key]
            if key.startswith('fpn.lat_layers'):
                state_dict[key.replace('fpn.', 'fpn_phase_1.')] = state_dict[key]
                del state_dict[key]
            elif key.startswith('fpn.') and key in state_dict:
                state_dict[key.replace('fpn.', 'fpn_phase_2.')] = state_dict[key]
                del state_dict[key]
        self.load_state_dict(state_dict)


class PredictionModule(nn.Module):
    """
    The (c) prediction module adapted from DSSD:
    https://arxiv.org/pdf/1701.06659.pdf

    Note that this is slightly different to the module in the paper
    because the Bottleneck block actually has a 3x3 convolution in
    the middle instead of a 1x1 convolution. Though, I really can't
    be arsed to implement it myself, and, who knows, this might be
    better.

    Args:
        - in_channels:   The input feature size.
        - out_channels:  The output feature size (must be a multiple of 4).
        - aspect_ratios: A list of lists of priorbox aspect ratios (one list per 
                         scale).
        - scales:        A list of priorbox scales relative to this layer's convsize.
                         For instance: If this layer has convouts of size 30x30 for
                         an image of size 600x600, the 'default' (scale of 1) for 
                         this layer would produce bounding boxes with an area of 
                         20x20px. If the scale is .5 on the other hand, this layer 
                         would consider bounding boxes with area 10x10px, etc.
        - parent:        If parent is a PredictionModule, this module will use 
                         all the layers from parent instead of from this module.
    """
    def __init__(self, 
                 in_channels, 
                 out_channels=1024, 
                 aspect_ratios=[[1]], 
                 scales=[1], 
                 parent=None, 
                 index=0):

        super().__init__()
        self.params = [in_channels, out_channels, aspect_ratios, scales, parent, index]
        self.num_classes = NUM_CLASSES
        self.mask_dim    = 32
        self.num_priors  = sum(len(x) for x in aspect_ratios)
        self.parent      = [parent]
        self.index       = index
        head_layer_params = {'kernel_size': 3, 'padding': 1}

        if parent is None:
            self.upfeature, out_channels = make_net(in_channels, [(256, 3, {'padding': 1})])
            self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4, **head_layer_params)
            self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.num_classes, **head_layer_params)
            self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, **head_layer_params)
            self.bbox_extra, self.conf_extra, self.mask_extra = [make_extra(x) for x in (0, 0, 0)]

        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.priors = None
        self.last_conv_size = None

    def forward(self, x):
        src = self if self.parent[0] is None else self.parent[0]
        conv_h = x.size(2)
        conv_w = x.size(3)

        try:
            x = src.upfeature(x)
        except:
            pass
            
        bbox_x = src.bbox_extra(x)
        conf_x = src.conf_extra(x)
        mask_x = src.mask_extra(x)

        bbox = src.bbox_layer(bbox_x).permute(
            0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        conf = src.conf_layer(conf_x).permute(
            0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        mask = src.mask_layer(mask_x).permute(
            0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
        
        mask = torch.tanh(mask)

        priors = self.make_priors(conv_h, conv_w)
        preds = {'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors}
        return preds
    
    def make_priors(self, conv_h, conv_w):
        if self.last_conv_size != (conv_w, conv_h):
            prior_data = []
            for j, i in product(range(conv_h), range(conv_w)):
                x = (i + 0.5) / conv_w
                y = (j + 0.5) / conv_h
                for scale, ars in zip(self.scales, self.aspect_ratios):
                    for ar in ars:
                        ar = sqrt(ar)
                        w = scale * ar / 550
                        h = w
                        prior_data += [x, y, w, h]
            self.priors = torch.Tensor(prior_data).view(-1, 4)
            self.last_conv_size = (conv_w, conv_h)
        return self.priors


class FPN_phase_1(ScriptModuleWrapper):
    __constants__ = ['interpolation_mode']
    def __init__(self, in_channels):
        super().__init__()
        self.src_channels = in_channels
        self.lat_layers = nn.ModuleList([
            nn.Conv2d(x, FPN_NUM_FEATURES, kernel_size=1)
            for x in reversed(in_channels)
        ])
        self.interpolation_mode = 'bilinear'

    def forward(self, x1=None, x2=None, x3=None, x4=None, x5=None, x6=None, x7=None):
        convouts_ = [x1, x2, x3, x4, x5, x6, x7]
        convouts = []
        j = 0
        while j < len(convouts_):
            t = convouts_[j]
            if t is not None:
                convouts.append(t)
            j += 1

        out = []
        lat_feats = []
        x = torch.zeros(1, device=convouts[0].device)

        for i in range(len(convouts)):
            out.append(x)
            lat_feats.append(x)

        j = len(convouts)
        for lat_layer in self.lat_layers:
            j -= 1
            if j < len(convouts) - 1:
                _, _, h, w = convouts[j].size()
                x = F.interpolate(x, size=(h, w), mode=self.interpolation_mode, 
                    align_corners=False)
            lat_j = lat_layer(convouts[j])
            lat_feats[j] = lat_j
            x = x + lat_j
            out[j] = x
            
        for i in range(len(convouts)):
            out.append(lat_feats[i])
        return out


class FPN_phase_2(ScriptModuleWrapper):
    __constants__ = ['num_downsample']
    def __init__(self, in_channels):
        super().__init__()
        self.src_channels = in_channels
        self.pred_layers = nn.ModuleList([
            nn.Conv2d(FPN_NUM_FEATURES, FPN_NUM_FEATURES, kernel_size=3, padding=1)
            for _ in in_channels])
        self.num_downsample = 2 
        self.downsample_layers = nn.ModuleList([
            nn.Conv2d(FPN_NUM_FEATURES, FPN_NUM_FEATURES, kernel_size=3, padding=1, stride=2)
            for _ in range(self.num_downsample)])

    def forward(self, x1=None, x2=None, x3=None, x4=None, x5=None, x6=None, x7=None) -> List[Tensor]:
        out_ = [x1, x2, x3, x4, x5, x6, x7]
        out = []
        j = 0
        while j < len(out_):
            t = out_[j]
            if t is not None:
                out.append(t)
            j += 1

        len_convouts = len(out)
        j = len_convouts
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = F.relu(pred_layer(out[j]))
        for downsample_layer in self.downsample_layers:
            out.append(downsample_layer(out[-1]))            
        return out