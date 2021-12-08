# Modifications copyright 2021 AI Singapore

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Original copyright (c) 2019 ZhongdaoWang

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

"""
Utilities functions
"""

from typing import Any, List, Tuple, Union
import cv2
import numpy as np
import torch
from torchvision.ops import nms

# pylint: disable=invalid-name, no-member


def letterbox(
    img: np.ndarray,
    height: int = 608,
    width: int = 1088,
    color: Tuple[float, float, float] = (127.5, 127.5, 127.5),
) -> Tuple[np.ndarray, float, float, float]:
    """Resize a rectangular image to a padded rectangular.

    Args:
        img (np.ndarray): Image frame
        height (int, optional): Height of padded image. Defaults to 608.
        width (int, optional): Width of padded image. Defaults to 1088.
        color (Tuple[float, float, float], optional): Colour of border
            around image. Defaults to (127.5, 127.5, 127.5).

    Returns:
        Tuple[np.ndarray, float, float, float]: Padded rectangular image.
    """
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (
        round(shape[1] * ratio),
        round(shape[0] * ratio),
    )  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # padded rectangular
    return img, ratio, dw, dh


def scale_coords(
    img_size: List[Any], coords: torch.Tensor, img0_shape: Tuple[Any, Any]
) -> torch.Tensor:
    """Rescale x1, y1, x2, y2 from 416 to image size.

    Args:
        img_size (List[Any]): Size of image.
        coords (torch.Tensor): Prediction detections.
        img0_shape (Tuple[Any, Any]): New image shape to scale to.

    Returns:
        torch.Tensor: Scaled coordinates.
    """
    gain_w = float(img_size[0]) / img0_shape[1]  # gain  = old / new
    gain_h = float(img_size[1]) / img0_shape[0]
    gain = min(gain_w, gain_h)
    pad_x = (img_size[0] - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size[1] - img0_shape[0] * gain) / 2  # height padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, 0:4] /= gain
    coords[:, :4] = torch.clamp(coords[:, :4], min=0)
    return coords


def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.5,
    nms_thres: float = 0.4,
) -> Union[List[torch.Tensor], List[None]]:
    """Removes detections with lower object confidence score than 'conf_thres'.
    Non-Maximum Suppression to further filter detections. This algorithm
    has been hardcoded to use the 'standard' metric for nms.

    Args:
        prediction (torch.Tensor): Input image frame.
        conf_thres (float): Confidence threshold value.
        nms_thres (float): NMS threshold value.

    Returns:
        List[torch.Tensor | None]: Returns detections with shape;
            (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    output = [None for _ in range(len(prediction))]
    for image_i, pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        # Get score and class with highest confidence

        v = pred[:, 4] > conf_thres
        v = v.nonzero().squeeze()
        if len(v.shape) == 0:
            v = v.unsqueeze(0)

        pred = pred[v]

        # If none are remaining => process next image
        nP = pred.shape[0]
        if not nP:
            continue
        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])

        # Non-maximum suppression
        nms_indices = nms(pred[:, :4], pred[:, 4], nms_thres)
        det_max = pred[nms_indices]

        if len(det_max) > 0:
            # Add max detections to outputs
            output[image_i] = (
                det_max
                if output[image_i] is None
                else torch.cat((output[image_i], det_max))  # type: ignore
            )
    return output


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2].
    (x, y) are coordinates of center. (x1, y1) and (x2, y2) are coordinates
    of bottom left and top right respectively.

    Args:
        x (torch.Tensor): Tensor with bounding boxes of format
            `(center x, center y, width, height)`.

    Returns:
        torch.Tensor: Tensor with bounding boxes of format
            `(top left x, top left y, bottom right x, bottom right y)`.
    """
    y = torch.zeros_like(x) if x.dtype is torch.float32 else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # Bottom left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # Bottom left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # Top right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # Top right y
    return y


def generate_anchor(
    nGh: int, nGw: int, anchor_wh: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Generate anchor bounding box for prediction.

    Args:
        nGh (int): Number of grid height.
        nGw (int): Number of grid width.
        anchor_wh (torch.Tensor): Anchor for bounding boc width, height.
        device (torch.device): Device type being used. "cuda" or "cpu".

    Returns:
        torch.Tensor: Prediction bounding box anchors.
    """
    nA = len(anchor_wh)
    yy, xx = torch.meshgrid(torch.arange(nGh), torch.arange(nGw))
    xx = xx.to(device)
    yy = yy.to(device)

    mesh = torch.stack([xx, yy], dim=0)  # Shape 2, nGh, nGw
    mesh = mesh.unsqueeze(0).repeat(nA, 1, 1, 1).float()  # Shape nA x 2 x nGh x nGw
    anchor_offset_mesh = (
        anchor_wh.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, nGh, nGw)
    )  # Shape nA x 2 x nGh x nGw
    anchor_mesh = torch.cat(
        [mesh, anchor_offset_mesh], dim=1
    )  # Shape nA x 4 x nGh x nGw
    return anchor_mesh


def decode_delta(delta: torch.Tensor, fg_anchor_list: torch.Tensor) -> torch.Tensor:
    """Decode delta tensors.

    Args:
        delta (torch.Tensor): Detection prediction tensor.
        fg_anchor_list (torch.Tensor): Anchors for bounding box.

    Returns:
        torch.Tensor: Prediction tensors.
    """
    px, py, pw, ph = (
        fg_anchor_list[:, 0],
        fg_anchor_list[:, 1],
        fg_anchor_list[:, 2],
        fg_anchor_list[:, 3],
    )
    dx, dy, dw, dh = delta[:, 0], delta[:, 1], delta[:, 2], delta[:, 3]
    gx = pw * dx + px
    gy = ph * dy + py
    gw = pw * torch.exp(dw)
    gh = ph * torch.exp(dh)
    return torch.stack([gx, gy, gw, gh], dim=1)


def decode_delta_map(
    delta_map: torch.Tensor,
    anchors: Union[torch.Tensor, torch.nn.Module],
    device: torch.device,
) -> torch.Tensor:
    """Decode delta map.

    Args:
        delta_map (torch.Tensor): shape (nB, nA, nGh, nGw, 4).
        anchors (torch.Tensor): shape (nA,4).

    Returns:
        torch.Tensor: Prediction map.
    """
    nB, nA, nGh, nGw, _ = delta_map.shape
    anchor_mesh = generate_anchor(nGh, nGw, anchors, device)  # type: ignore
    anchor_mesh = anchor_mesh.permute(
        0, 2, 3, 1
    ).contiguous()  # Shpae (nA x nGh x nGw) x 4
    anchor_mesh = anchor_mesh.unsqueeze(0).repeat(nB, 1, 1, 1, 1)
    pred_list = decode_delta(delta_map.view(-1, 4), anchor_mesh.view(-1, 4))
    pred_map = pred_list.view(nB, nA, nGh, nGw, 4)
    return pred_map
