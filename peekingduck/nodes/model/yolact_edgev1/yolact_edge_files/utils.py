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

from typing import Any, Callable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FastBaseTransform(torch.nn.Module):
    """
    Transform that does all operations on the GPU for improved speed.
    Maintain this as necessary.
    """

    def __init__(self, input_size: Tuple[int, int]) -> None:
        super().__init__()
        self.input_size = input_size
        # The tensor values are in BGR and are the means and standard deviation
        # values for ImageNet respectively
        if torch.cuda.is_available():
            self.mean = (
                Tensor((103.94, 116.78, 123.68)).float().cuda()[None, :, None, None]
            )
            self.std = Tensor((57.38, 57.12, 58.40)).float().cuda()[None, :, None, None]
        else:
            self.mean = Tensor((103.94, 116.78, 123.68)).float()[None, :, None, None]
            self.std = Tensor((57.38, 57.12, 58.40)).float()[None, :, None, None]

    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (Tensor): input image of shape (N, H, W, C)

        Returns:
            img (Tensor): transformed image of shape (N, C, H, W) where
                H == W
        """
        self.mean = self.mean.to(img.device)
        self.std = self.std.to(img.device)

        img = img.permute(0, 3, 1, 2).contiguous()
        img = F.interpolate(img, self.input_size, mode="bilinear", align_corners=False)

        img = (img - self.mean) / self.std
        img = img[:, (2, 1, 0), :, :].contiguous()
        return img


class InterpolateModule(nn.Module):
    """
    This is a module version of F.interpolate.
    Any arguments you give it just get passed along for the ride.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs: A tensor of shape (N, C, H, W)

        Returns:
            A tensor of shape (N, C, H', W')
        """
        return F.interpolate(inputs, *self.args, **self.kwargs)


def make_net(
    in_channels: int,
    conf: List[Any],
    include_last_relu: bool = True,
) -> Tuple[nn.Sequential, int]:
    """
    A helper function to take a config setting and turn it into a network.
    Used by protonet and extrahead. Returns (network, out_channels)

    Args:
        in_channels (int): number of channels in the input
        conf (List[Any]): a list of configs to create the network
    """

    def make_layer(layer_config: Tuple[int, int, dict]) -> List[Callable]:
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

        in_channels = num_channels if num_channels is not None else in_channels
        return [layer, nn.ReLU(inplace=True)]

    net: List[Any] = sum([make_layer(x) for x in conf], [])
    if not include_last_relu:
        net = net[:-1]

    return nn.Sequential(*(net)), in_channels


def make_extra(num_layers: int, out_channels: int) -> Callable:
    """
    Lambda function that creates an array of num_layers alternating conv-relu if
    the num_layers is not 0.

    Args:
        num_layers (int): number of layers to create
        out_channels (int): number of channels in the output

    Returns:
        An array of alternating convolutional ReLU layers if there is at least
        one layer.
    """
    if num_layers == 0:
        return lambda x: x
    return nn.Sequential(
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


def jaccard(box_a: Tensor, box_b: Tensor, is_crowd: bool = False) -> Tensor:
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)

    Args:
        box_a (Tensor): Ground truth bounding boxes, Shape: [num_objects,4]
        box_b (Tensor): Prior boxes from priorbox layers, Shape: [num_priors,4]

    Returns:
        out (Tensor): Shape: [box_a.size(0), box_b.size(0)]
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
    out = inter / area_a if is_crowd else inter / union
    return out if use_batch else out.squeeze(0)


def point_form(boxes: Tensor) -> Tensor:
    """Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.

    Args:
        boxes: (Tensor) center-size default boxes from priorbox layers.

    Returns:
        (Tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat(
        (
            boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
            boxes[:, :2] + boxes[:, 2:] / 2,
        ),
        1,
    )  # xmax, ymax


def intersect(box_a: Tensor, box_b: Tensor) -> Tensor:
    """Resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.

    Args:
        box_a: (Tensor) bounding boxes, Shape: [n,A,4].
        box_b: (Tensor) bounding boxes, Shape: [n,B,4].

    Returns:
        (Tensor) intersection area, Shape: [n,A,B].
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


def decode(loc: Tensor, priors: Tensor, use_yolo_regressors: bool = False) -> Tensor:
    """
    Decode predicted bbox coordinates using the same scheme
    employed by Yolov2: https://arxiv.org/pdf/1612.08242.pdf
        b_x = (sigmoid(pred_x) - .5) / conv_w + prior_x
        b_y = (sigmoid(pred_y) - .5) / conv_h + prior_y
        b_w = prior_w * exp(loc_w)
        b_h = prior_h * exp(loc_h)

    Note that loc is inputed as [(s(x)-.5)/conv_w, (s(y)-.5)/conv_h, w, h]
    while priors are inputed as [x, y, w, h] where each coordinate
    is relative to size of the image (even sigmoid(x)). We do this
    in the network by dividing by the 'cell size', which is just
    the size of the convouts.

    Also note that prior_x and prior_y are center coordinates which
    is why we have to subtract .5 from sigmoid(pred_x and pred_y).

    Args:
        loc (Tensor): The predicted bounding boxes of size [num_priors, 4]
        priors (Tensor): The priorbox coords with size [num_priors, 4]
        use_yolo_regressors (bool): Whether or not to use the YOLO regressors.

    Returns:
        boxes (Tensor): A tensor of decoded relative coordinates in point form
            form with size [num_priors, 4]
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


def sanitize_coordinates(
    _x1: Tensor, _x2: Tensor, img_size: int, padding: int = 0, cast: bool = True
) -> Tuple[Tensor, Tensor]:
    """Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0,
    and x2 <= image_size. Also converts from relative to absolute coordinates and
    casts the results to long tensors. If cast is false, the result won't be cast
    to longs.

    Args:
        _x1 (Tensor): input coordinate of x1.
        _x2 (Tensor): input coordinate of x2.
        img_size (int): the dimensions of the image.
        padding (int, optional): Padding value. Defaults to 0.
        cast (bool, optional): Cast the results to long tensors. Defaults to True.

    Returns:
        x_1, x_2: The sanitized input coordinates of _x1 and _x2 respectively.
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


def crop(  # pylint: disable=too-many-locals
    masks: Tensor, boxes: Tensor, padding: int = 1
) -> Tensor:
    """ "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong Zhou.

    Args:
        masks (Tensor): Uncropped mask tensor values in uint8
        boxes (Tensor): x1, y1, x2, y2 values of the bounding box detection
        padding (int, optional): Padding value. Defaults to 1.

    Returns:
        out (Tensor): cropped mask tensor
    """
    height, width, batch = masks.size()
    x_1, x_2 = sanitize_coordinates(
        boxes[:, 0], boxes[:, 2], width, padding, cast=False
    )
    y_1, y_2 = sanitize_coordinates(
        boxes[:, 1], boxes[:, 3], height, padding, cast=False
    )

    rows = (
        torch.arange(width, device=masks.device, dtype=x_1.dtype)
        .view(1, -1, 1)
        .expand(height, width, batch)
    )

    cols = (
        torch.arange(height, device=masks.device, dtype=x_1.dtype)
        .view(-1, 1, 1)
        .expand(height, width, batch)
    )

    masks_left = rows >= x_1.view(1, 1, -1)
    masks_right = rows < x_2.view(1, 1, -1)
    masks_up = cols >= y_1.view(1, 1, -1)
    masks_down = cols < y_2.view(1, 1, -1)
    crop_mask = masks_left * masks_right * masks_up * masks_down
    out = masks * crop_mask.float()
    return out
