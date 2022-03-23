# Modifications copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Original copyright (c) 2019 ZhongdaoWang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Utility functions for JDE model.

Modifications:
- Removed unused ratio, width and height padding return values in letterbox()
- Removed filtering detections by score_threshold in non_max_suppression since
    it's already performed in track.py before calling non_max_suppression
"""

from typing import List, Tuple

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from peekingduck.pipeline.utils.bbox.transforms import xywh2xyxy


def decode_delta(delta: torch.Tensor, anchor_mesh: torch.Tensor) -> torch.Tensor:
    """Converts raw output to (x, y, w, h) format where (x, y) is the center,
    w is the width, and h is the height of the bounding box.

    Args:
        delta (torch.Tensor): Raw output from the YOLOLayer.
        anchor_mesh (torch.Tensor): Tensor containing the grid points and their
            respective anchor offsets.

    Returns:
        (torch.Tensor): Decoded bounding box tensor.
    """
    delta[..., :2] = delta[..., :2] * anchor_mesh[..., 2:] + anchor_mesh[..., :2]
    delta[..., 2:] = torch.exp(delta[..., 2:]) * anchor_mesh[..., 2:]
    return delta


def decode_delta_map(
    delta_map: torch.Tensor, anchors: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Decodes raw bounding box output in to (x, y, w, h) format where
    (x, y) is the center, w is the width, and h is the height.

    Args:
        delta_map (torch.Tensor): A tensor with the shape
            (batch_size, num_anchors, grid_height, grid_width, 4) containing
            raw bounding box predictions.
        anchors (torch.Tensor): A tensor with the shape (num_anchors, 4)
            containing the anchors used for the `YOLOLayer`.
        device (torch.device): The device which a `torch.Tensor` is on or
            will be allocated.

    Returns:
        (torch.Tensor): Tensor containing the decoded bounding boxes.
    """
    batch_size, num_anchors, grid_height, grid_width, _ = delta_map.shape
    anchor_mesh = generate_anchor(grid_height, grid_width, anchors, device)
    # Shape (num_anchors x grid_height x grid_width) x 4
    anchor_mesh = anchor_mesh.permute(0, 2, 3, 1).contiguous()
    anchor_mesh = anchor_mesh.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    pred_list = decode_delta(delta_map.view(-1, 4), anchor_mesh.view(-1, 4))
    pred_map = pred_list.view(batch_size, num_anchors, grid_height, grid_width, 4)
    return pred_map


def generate_anchor(
    grid_height: int, grid_width: int, anchor_wh: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Generates grid anchors for a single level.

    Args:
        grid_height (int): Height of feature map.
        grid_width (int): Width of feature map.
        anchor_wh (torch.Tensor): Width and height of the anchor boxes.
        device (torch.device): The device which a `torch.Tensor` is on or
            will be allocated.

    Returns:
        (torch.Tensor): Anchors of a feature map in a single level.
    """
    num_anchors = len(anchor_wh)
    y_vec, x_vec = torch.meshgrid(torch.arange(grid_height), torch.arange(grid_width))
    x_vec, y_vec = x_vec.to(device), y_vec.to(device)

    # Shape 2 x grid_height x grid_width
    mesh = torch.stack([x_vec, y_vec], dim=0)
    # Shape num_anchors x 2 x grid_height x grid_width
    mesh = mesh.unsqueeze(0).repeat(num_anchors, 1, 1, 1).float()
    # Shape num_anchors x 2 x grid_height x grid_width
    anchor_offset_mesh = (
        anchor_wh.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, grid_height, grid_width)
    )
    # Shape num_anchors x 4 x grid_height x grid_width
    anchor_mesh = torch.cat([mesh, anchor_offset_mesh], dim=1)
    return anchor_mesh


def letterbox(
    image: np.ndarray,
    height: int,
    width: int,
    color: Tuple[float, float, float] = (127.5, 127.5, 127.5),
) -> np.ndarray:
    """Resizes a rectangular image to a padded rectangular image.

    Args:
        image (np.ndarray): Image frame.
        height (int): Height of padded image.
        width (int): Width of padded image.
        color (Tuple[float, float, float]): Color used for padding around
            the image. (127.5, 127.5, 127.5) is chosen as it is used by the
            original project during model training.

    Returns:
        (np.ndarray): Padded rectangular image.
    """
    shape = image.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    # new_shape = [width, height]
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    width_padding = (width - new_shape[0]) / 2
    height_padding = (height - new_shape[1]) / 2
    top = round(height_padding - 0.1)
    bottom = round(height_padding + 0.1)
    left = round(width_padding - 0.1)
    right = round(width_padding + 0.1)
    # resized, no border
    image = cv2.resize(image, new_shape, interpolation=cv2.INTER_AREA)
    # padded rectangular
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return image


def non_max_suppression(
    prediction: torch.Tensor, nms_threshold: float
) -> List[torch.Tensor]:
    """Removes detections with lower object confidence score than
    `score_threshold`. Non-Maximum Suppression to further filter detections.

    Args:
        prediction (torch.Tensor): Predicted bounding boxes.
        nms_threshold (float): Threshold for Intersection-over-Union values of
            the bounding boxes.

    Returns:
        (List[Optional[torch.Tensor]]): List of detections with shape
            (x1, y1, x2, y2, object_conf, class_score, class_pred). For
            detections which have all bounding boxes filtered by `nms`, the
            element will be `None` instead.
    """
    # Initializing this list with torch.empty will likely incur some additional
    # computational cost
    output: List[torch.Tensor] = [None for _ in range(len(prediction))]  # type: ignore
    for i, pred in enumerate(prediction):
        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])
        # Non-maximum suppression
        nms_indices = nms(pred[:, :4], pred[:, 4], nms_threshold)
        det_max = pred[nms_indices]

        if len(det_max) == 0:  # pragma: no cover
            # This really shouldn't happen since nms will at worst leave one
            # bbox and suppress everything else
            continue
        # Add max detections to outputs
        output[i] = det_max if output[i] is None else torch.cat((output[i], det_max))

    return output


def scale_coords(
    img_size: List[int], coords: torch.Tensor, img0_size: Tuple[int, int]
) -> torch.Tensor:
    """Rescales bounding box coordinates (x1, y1, x2, y2) from `img_size` to
    `img0_size`.

    Args:
        img_size (List[int]): Model input size (w x h).
        coords (torch.Tensor): Bounding box coordinates.
        img0_size (Tuple[int, int]): Size of original video frame (h x w).

    Returns:
        (torch.Tensor): Bounding boxes with resized coordinates.
    """
    # gain = old / new
    gain = min(float(img_size[0]) / img0_size[1], float(img_size[1]) / img0_size[0])
    pad_x = (img_size[0] - img0_size[1] * gain) / 2  # width padding
    pad_y = (img_size[1] - img0_size[0] * gain) / 2  # height padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, :4] /= gain
    coords[:, :4] = torch.clamp(coords[:, :4], min=0)
    return coords
