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
- Modified docstrings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastBaseTransform(torch.nn.Module):
    """
    Transform that does all operations on the GPU for improved speed.
    This doesn't suppport a lot of configs and should only be used for production.
    Maintain this as necessary.
    """

    def __init__(self):
        super().__init__()
        if torch.cuda.is_available():
            self.mean = (
                torch.Tensor((103.94, 116.78, 123.68))
                .float()
                .cuda()[None, :, None, None]
            )
            self.std = (
                torch.Tensor((57.38, 57.12, 58.40)).float().cuda()[None, :, None, None]
            )
        else:
            self.mean = torch.Tensor((103.94, 116.78, 123.68)).float()[
                None, :, None, None
            ]
            self.std = torch.Tensor((57.38, 57.12, 58.40)).float()[None, :, None, None]

    def forward(self, img):
        """
        Args:
            img (torch.Tensor): input image of shape (N, H, W, C)

        Returns:
            img (torch.Tensor): transformed image of shape (N, C, H, W) where
                H == W
        """
        self.mean = self.mean.to(img.device)
        self.std = self.std.to(img.device)

        img = img.permute(0, 3, 1, 2).contiguous()
        img = F.interpolate(img, (550, 550), mode="bilinear", align_corners=False)

        img = (img - self.mean) / self.std
        img = img[:, (2, 1, 0), :, :].contiguous()
        return img


class InterpolateModule(nn.Module):
    """
    This is a module version of F.interpolate.
    Any arguments you give it just get passed along for the ride.
    """

    def __init__(self, *args, **kwdargs):
        super().__init__()
        self.args = args
        self.kwdargs = kwdargs

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: A tensor of shape (N, C, H, W)
        Returns:
            A tensor of shape (N, C, H', W')
        """
        return F.interpolate(inputs, *self.args, **self.kwdargs)


def make_net(in_channels, conf, include_last_relu=True):
    """
    A helper function to take a config setting and turn it into a network.
    Used by protonet and extrahead. Returns (network, out_channels)
    """

    def make_layer(layer_config):
        nonlocal in_channels
        num_channels = layer_config[0]
        kernel_size = layer_config[1]
        if kernel_size > 0:
            layer = nn.Conv2d(in_channels, num_channels, kernel_size, **layer_config[2])
        else:
            if num_channels is None:
                layer = InterpolateModule(
                    scale_factor=-kernel_size,
                    mode="bilinear",
                    align_corners=False,
                    **layer_config[2],
                )
            else:
                layer = nn.ConvTranspose2d(
                    in_channels, num_channels, -kernel_size, **layer_config[2]
                )

        in_channels = num_channels if num_channels is not None else in_channels
        return [layer, nn.ReLU(inplace=True)]

    net = sum([make_layer(x) for x in conf], [])
    if not include_last_relu:
        net = net[:-1]
    return nn.Sequential(*(net)), in_channels


def make_extra(num_layers, out_channels):
    """
    Lambda function that creates an array of num_layers alternating conv-relu if
    the num_layers is not 0.
    """
    if num_layers == 0:
        out = lambda x: x

    else:
        out = nn.Sequential(
            *sum(
                [
                    [
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                    ]
                    for _ in range(num_layers)
                ],
                [],
            )
        )
    return out


def jaccard(box_a, box_b, iscrowd=False):
    """
    Compute the jaccard overlap of two sets of boxes.
    """
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]
    inter = intersect(box_a, box_b)
    area_a = (
        ((box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1]))
        .unsqueeze(2)
        .expand_as(inter)
    )
    area_b = (
        ((box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1]))
        .unsqueeze(1)
        .expand_as(inter)
    )
    union = area_a + area_b - inter
    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)


def point_form(boxes):
    """Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.

    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.

    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat(
        (
            boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
            boxes[:, :2] + boxes[:, 2:] / 2,
        ),
        1,
    )  # xmax, ymax


def intersect(box_a, box_b):
    """
    Compute intersection between two sets of boxes.
    """
    num = box_a.size(0)
    set_a = box_a.size(1)
    set_b = box_b.size(1)

    max_xy = torch.min(
        box_a[:, :, 2:].unsqueeze(2).expand(num, set_a, set_b, 2),
        box_b[:, :, 2:].unsqueeze(1).expand(num, set_a, set_b, 2),
    )
    min_xy = torch.max(
        box_a[:, :, :2].unsqueeze(2).expand(num, set_a, set_b, 2),
        box_b[:, :, :2].unsqueeze(1).expand(num, set_a, set_b, 2),
    )
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, :, 0] * inter[:, :, :, 1]


def decode(loc, priors, use_yolo_regressors: bool = False):
    """
    Decode predicted bbox locations from locations and priors.
    """
    if use_yolo_regressors:
        boxes = torch.cat(
            (loc[:, :2] + priors[:, :2], priors[:, 2:] * torch.exp(loc[:, 2:])), 1
        )
        boxes = point_form(boxes)
    else:
        variances = [0.1, 0.2]
        boxes = torch.cat(
            (
                priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
            ),
            1,
        )
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
    return boxes


def sanitize_coordinates(_x1, _x2, img_size: int, padding: int = 0, cast: bool = True):
    """
    Sanitize the coordinates to be within [0, img_size-1].
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    if cast:
        _x1 = _x1.long()
        _x2 = _x2.long()
    x_1 = torch.min(_x1, _x2)
    x_2 = torch.max(_x1, _x2)
    x_1 = torch.clamp(x_1 - padding, min=0)
    x_2 = torch.clamp(x_2 + padding, max=img_size)
    return x_1, x_2
