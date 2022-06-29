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

"""A collection of utility functions used by YolactEdge

Modifications include:
- Removed unused python scripts in the original utils folder
- Removed unused utility functions from the original repository
- Merged utility functions from the layers folder
- Refactored config file parsing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import pickle as pkl

from collections import defaultdict

SELECTED_LAYERS = list(range(1,4)) # ResNet 50/101 FPN backbone
# SELECTED_LAYERS = [3, 4, 6] # MobileNetV2

SRC_CHANNELS = [256, 512, 1024, 2048]
SCORE_THRESHOLD = 0.1
TOP_K = 15 # Change for top number of detections

NUM_DOWNSAMPLE = 2
FPN_NUM_FEATURES = 256
NUM_CLASSES = 81
MAX_NUM_DETECTIONS = 100

# Can be removed for PKD implementation
COLORS = ((244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 193,   7),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          (158, 158, 158),
          ( 96, 125, 139))

# Can be removed for PKD implementation
COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard','sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

class FastBaseTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.mean = torch.Tensor((103.94, 116.78, 123.68)).float().cuda()[None, :, None, None]
            self.std  = torch.Tensor((57.38, 57.12, 58.40)).float().cuda()[None, :, None, None]
        except:
            self.mean = torch.Tensor((103.94, 116.78, 123.68)).float()[None, :, None, None]
            self.std  = torch.Tensor((57.38, 57.12, 58.40)).float()[None, :, None, None]

    def forward(self, img):
        self.mean = self.mean.to(img.device)
        self.std  = self.std.to(img.device)

        img = img.permute(0, 3, 1, 2).contiguous()
        img = F.interpolate(img, (550, 550), mode='bilinear', align_corners=False)

        img = (img - self.mean) / self.std
        img = img[:, (2, 1, 0), :, :].contiguous()
        return img


class InterpolateModule(nn.Module):
	def __init__(self, *args, **kwdargs):
		super().__init__()
		self.args = args
		self.kwdargs = kwdargs

	def forward(self, x):
		return F.interpolate(x, *self.args, **self.kwdargs)

def make_net(in_channels, conf, include_last_relu=True):
    def make_layer(layer_config):
        nonlocal in_channels
        num_channels = layer_config[0]
        kernel_size = layer_config[1]
        if kernel_size > 0:
            layer = nn.Conv2d(in_channels, num_channels, kernel_size, **layer_config[2])
        else:
            if num_channels is None:
                layer = InterpolateModule(scale_factor=-kernel_size, 
                                          mode='bilinear', 
                                          align_corners=False, 
                                          **layer_config[2])
            else:
                layer = nn.ConvTranspose2d(
                    in_channels, 
                    num_channels, 
                    -kernel_size, 
                    **layer_config[2]
                )

        in_channels = num_channels if num_channels is not None else in_channels
        return [layer, nn.ReLU(inplace=True)]

    net = sum([make_layer(x) for x in conf], [])
    if not include_last_relu:
        net = net[:-1]
    return nn.Sequential(*(net)), in_channels

def jaccard(box_a, box_b, iscrowd=False):
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2]-box_a[:, :, 0]) *
              (box_a[:, :, 3]-box_a[:, :, 1])).unsqueeze(2).expand_as(inter)
    area_b = ((box_b[:, :, 2]-box_b[:, :, 0]) *
              (box_b[:, :, 3]-box_b[:, :, 1])).unsqueeze(1).expand_as(inter)
    union = area_a + area_b - inter
    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)

@torch.jit.script
def point_form(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax

@torch.jit.script
def intersect(box_a, box_b):
    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, :, 0] * inter[:, :, :, 1]

@torch.jit.script
def decode(loc, priors, use_yolo_regressors:bool=False):
    if use_yolo_regressors:
        boxes = torch.cat((
            loc[:, :2] + priors[:, :2],
            priors[:, 2:] * torch.exp(loc[:, 2:])
        ), 1)
        boxes = point_form(boxes)
    else:
        variances = [0.1, 0.2] 
        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
    return boxes

@torch.jit.script
def sanitize_coordinates(_x1, _x2, img_size:int, padding:int=0, cast:bool=True):
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    if cast:
        _x1 = _x1.long()
        _x2 = _x2.long()
    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1-padding, min=0)
    x2 = torch.clamp(x2+padding, max=img_size)
    return x1, x2

@torch.jit.script
def crop(masks, boxes, padding:int=1):
    h, w, n = masks.size()
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=False)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=False)

    rows = torch.arange(w, 
        device=masks.device,
        dtype=x1.dtype
        ).view(1, -1, 1).expand(h, w, n)

    cols = torch.arange(h, 
        device=masks.device,
        dtype=x1.dtype
        ).view(-1, 1, 1).expand(h, w, n)

    masks_left  = rows >= x1.view(1, 1, -1)
    masks_right = rows <  x2.view(1, 1, -1)
    masks_up    = cols >= y1.view(1, 1, -1)
    masks_down  = cols <  y2.view(1, 1, -1)
    crop_mask = masks_left * masks_right * masks_up * masks_down
    return masks * crop_mask.float()

#TODO: Move to Detector object
def postprocess(det_output, w, h, batch_idx=0, interpolation_mode='bilinear',
                crop_masks=True, score_threshold=0):
    dets = det_output[batch_idx]
    if dets is None:
        return [torch.Tensor()] * 4
    if score_threshold > 0:
        keep = dets['score'] > score_threshold
        for k in dets:
            if k != 'proto':
                dets[k] = dets[k][keep]
        if dets['score'].size(0) == 0:
            return [torch.Tensor()] * 4

    b_w, b_h = (w, h)

    classes = dets['class']
    boxes   = dets['box']
    scores  = dets['score']
    masks   = dets['mask']
    proto_data = dets['proto']

    masks = proto_data @ masks.t()
    masks = torch.sigmoid(masks)

    boxes_np = boxes.cpu().detach().numpy()

    #TODO: Remove for final PeekingDuck implementation
    with open('bboxes-r101.pkl', 'wb') as f:
        pkl.dump(boxes_np, f)

    if crop_masks:
        masks = crop(masks, boxes)

    scores_np = scores.cpu().detach().numpy()

    #TODO: Remove for final PeekingDuck implementation
    with open('bbox_scores-r101.pkl', 'wb') as f:
        pkl.dump(scores_np, f)

    masks = masks.permute(2, 0, 1).contiguous()
    masks = F.interpolate(masks.unsqueeze(0), (h, w), 
        mode=interpolation_mode, align_corners=False).squeeze(0)
    masks.gt_(0.5)
    
    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(
        boxes[:, 0], boxes[:, 2], b_w, cast=False)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(
        boxes[:, 1], boxes[:, 3], b_h, cast=False)
    boxes = boxes.long()

    masks_np = masks.cpu().detach().numpy()

    #TODO: Remove for final PeekingDuck implementation
    with open('masks.pkl-r101', 'wb') as f:
        pkl.dump(masks_np, f)

    return classes, scores, boxes, masks

#TODO: Remove for final PeekingDuck implementation
def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.9):
    img_gpu = img / 255.0
    h, w, _ = img.shape

    def get_color(j, on_gpu=None):
        color_cache = defaultdict(lambda: {})
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    t = postprocess(dets_out, w, h, score_threshold=SCORE_THRESHOLD)
    try:
        torch.cuda.synchronize()
    except:
        pass

    masks = t[3][:TOP_K]
    classes, scores, boxes = [x[:TOP_K].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(TOP_K, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < SCORE_THRESHOLD:
            num_dets_to_consider = j
            break
    if num_dets_to_consider == 0:
        return (img_gpu * 255).byte().cpu().numpy()

    # This will only work on CUDA machines
    try: 
        masks = masks[:num_dets_to_consider, :, :, None]
        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha
        inv_alph_masks = masks * (-mask_alpha) + 1
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)
        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand 
    except:
        pass

    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    classes_np = []
    for j in reversed(range(num_dets_to_consider)):
        x1, y1, x2, y2 = boxes[j, :]
        color = get_color(j)
        score = scores[j]
        cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)
        _class = COCO_CLASSES[classes[j]] # change this to suit pkd
        text_str = '%s: %.2f' % (_class, score)
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        font_thickness = 1
        text_w, text_h = cv2.getTextSize(text_str, 
                                         font_face, 
                                         font_scale, 
                                         font_thickness)[0]
        text_pt = (x1, y1 - 3)
        text_color = [255, 255, 255]
        cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
        cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, 
            text_color, font_thickness, cv2.LINE_AA)
        classes_np.append(_class)

    classes_np.reverse()
    classes_np = np.asarray(classes_np)

    #TODO: Remove for final PeekingDuck implementation
    with open('bbox_labels-r101.pkl', 'wb') as f:
        pkl.dump(classes_np, f)
        
    return img_numpy