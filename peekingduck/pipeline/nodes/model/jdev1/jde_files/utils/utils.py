# Modifications copyright 2021 AI Singapore
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

"""
Utilities functions.
"""

from typing import List, Tuple, Union
import cv2
import numpy as np
import torch
from torchvision.ops import nms


def letterbox(
    img: np.ndarray,
    height: int = 608,
    width: int = 1088,
    color: Tuple[float, float, float] = (127.5, 127.5, 127.5),
) -> Tuple[np.ndarray, float, float, float]:
    """Resize a rectangular image to a padded rectangular.

    Args:
        img (np.ndarray): Image frame.
        height (int): Height of padded image. Defaults to 608.
        width (int): Width of padded image. Defaults to 1088.
        color (Tuple[float, float, float]): Colour of border around
            image, this colour is used as it is a midpoint between the
            range of pixel colours. The original project used this colour
            for data augmentation during model training. Defaults to
            (127.5, 127.5, 127.5).

    Returns:
        Tuple[np.ndarray, float, float, float]: Padded rectangular image.
    """
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    # new_shape = [width, height]
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    width_padding = (width - new_shape[0]) / 2
    height_padding = (height - new_shape[1]) / 2
    top, bottom = round(height_padding - 0.1), round(height_padding + 0.1)
    left, right = round(width_padding - 0.1), round(width_padding + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # padded rectangular
    return img, ratio, width_padding, height_padding


def scale_coords(
    img_size: List[int], coords: torch.Tensor, img0_shape: Tuple[int, int]
) -> torch.Tensor:
    """Rescale x1, y1, x2, y2 from 416 to image size.

    Args:
        img_size (List[int]): Size of image.
        coords (torch.Tensor): Prediction detections.
        img0_shape (Tuple[int, int]): New image shape to scale to.

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

        scores = pred[:, 4] > conf_thres
        scores = scores.nonzero().squeeze()
        if len(scores.shape) == 0:
            scores = scores.unsqueeze(0)

        pred = pred[scores]

        # If none are remaining => process next image
        num_pred = pred.shape[0]
        if not num_pred:
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


def xywh2xyxy(inputs: torch.Tensor) -> torch.Tensor:
    """Converts format of a bounding box. From [x, y, w, h] to
    [x1, y1, x2, y2], where (x, y) are coordinates of center and (x1, y1)
    and (x2, y2) are coordinates of the bottom left and the top right
    points respectively.

    Args:
        inputs (torch.Tensor): Tensor with bounding boxes of format
            `(center x, center y, width, height)`.

    Returns:
        torch.Tensor: Tensor with bounding boxes of format
            `(top left x, top left y, bottom right x, bottom right y)`.
    """
    outputs = (
        torch.zeros_like(inputs)
        if inputs.dtype is torch.float32
        else np.zeros_like(inputs)
    )
    outputs[:, 0] = inputs[:, 0] - inputs[:, 2] / 2  # Bottom left x
    outputs[:, 1] = inputs[:, 1] - inputs[:, 3] / 2  # Bottom left y
    outputs[:, 2] = inputs[:, 0] + inputs[:, 2] / 2  # Top right x
    outputs[:, 3] = inputs[:, 1] + inputs[:, 3] / 2  # Top right y
    return outputs


def generate_anchor(
    num_grid_height: int,
    num_grid_width: int,
    anchor_wh: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Generate anchor bounding box for prediction.

    Args:
        num_grid_height (int): Number of grid height.
        num_grid_width (int): Number of grid width.
        anchor_wh (torch.Tensor): Anchor for bounding boc width, height.
        device (torch.device): Device type being used. "cuda" or "cpu".

    Returns:
        torch.Tensor: Prediction bounding box anchors.
    """
    num_anchor = len(anchor_wh)
    y_grid, x_grid = torch.meshgrid(
        torch.arange(num_grid_height), torch.arange(num_grid_width)
    )
    x_grid = x_grid.to(device)
    y_grid = y_grid.to(device)

    # Shape 2, num_grid_height, num_grid_width
    mesh = torch.stack([x_grid, y_grid], dim=0)
    # Shape num_anchor x 2 x num_grid_height x num_grid_width
    mesh = mesh.unsqueeze(0).repeat(num_anchor, 1, 1, 1).float()
    # Shape num_anchor x 2 x num_grid_height x num_grid_width
    anchor_offset_mesh = (
        anchor_wh.unsqueeze(-1)
        .unsqueeze(-1)
        .repeat(1, 1, num_grid_height, num_grid_width)
    )
    # Shape num_anchor x 4 x num_grid_height x num_grid_width
    anchor_mesh = torch.cat([mesh, anchor_offset_mesh], dim=1)
    return anchor_mesh


def decode_delta(delta: torch.Tensor, fg_anchor_list: torch.Tensor) -> torch.Tensor:
    """Decodes representation of detected bounding boxes.

    Args:
        delta (torch.Tensor): Detection prediction tensor.
        fg_anchor_list (torch.Tensor): Anchors for bounding box.

    Returns:
        torch.Tensor: Prediction tensors.
    """
    pixel_x, pixel_y, pixel_w, pixel_h = (
        fg_anchor_list[:, 0],
        fg_anchor_list[:, 1],
        fg_anchor_list[:, 2],
        fg_anchor_list[:, 3],
    )
    delta_x, delta_y, delta_width, delta_height = (
        delta[:, 0],
        delta[:, 1],
        delta[:, 2],
        delta[:, 3],
    )
    ground_x = pixel_w * delta_x + pixel_x
    ground_y = pixel_h * delta_y + pixel_y
    ground_w = pixel_w * torch.exp(delta_width)
    ground_h = pixel_h * torch.exp(delta_height)
    return torch.stack([ground_x, ground_y, ground_w, ground_h], dim=1)


def decode_delta_map(
    delta_map: torch.Tensor,
    anchors: Union[torch.Tensor, torch.nn.Module],
    device: torch.device,
) -> torch.Tensor:
    """Decodes YOLO model output into useable predictions.

    Args:
        delta_map (torch.Tensor): shape (nB, nA, nGh, nGw, 4).
        anchors (torch.Tensor): shape (nA,4).

    Returns:
        torch.Tensor: Readable prediction mapping.
    """
    num_boxes, num_anchor, num_grid_height, num_grid_width, _ = delta_map.shape
    anchor_mesh = generate_anchor(num_grid_height, num_grid_width, anchors, device)  # type: ignore
    # Shape (num_anchor x num_grid_height x num_grid_width) x 4
    anchor_mesh = anchor_mesh.permute(0, 2, 3, 1).contiguous()
    anchor_mesh = anchor_mesh.unsqueeze(0).repeat(num_boxes, 1, 1, 1, 1)
    pred_list = decode_delta(delta_map.view(-1, 4), anchor_mesh.view(-1, 4))
    pred_map = pred_list.view(num_boxes, num_anchor, num_grid_height, num_grid_width, 4)
    return pred_map


def bbox_overlap(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Calculate Intersection of Union between two sets of bounding
    boxes. Intersection over Union (IoU) of two bounding boxes A and B
    is calculated doing: (A ∩ B) / (A ∪ B).

    Args:
        bboxes1 (np.ndarray): Array of shape (total_bboxes1, 4).
        bboxes2 (np.ndarray): Array of shape (total_bboxes2, 4).

    Returns:
        np.ndarray: Array of shape (total_bboxes1, total_bboxes1) a
            matrix with the intersection over union of bboxes1[i] and
            bboxes2[j] in iou[i][j].
    """
    x_start = np.maximum(bboxes1[:, [0]], bboxes2[:, [0]].T)
    y_start = np.maximum(bboxes1[:, [1]], bboxes2[:, [1]].T)
    x_end = np.minimum(bboxes1[:, [2]], bboxes2[:, [2]].T)
    y_end = np.minimum(bboxes1[:, [3]], bboxes2[:, [3]].T)

    intersection = np.maximum(x_end - x_start + 1, 0.0) * np.maximum(
        y_end - y_start + 1, 0.0
    )

    bboxes1_area = (bboxes1[:, [2]] - bboxes1[:, [0]] + 1) * (
        bboxes1[:, [3]] - bboxes1[:, [1]] + 1
    )
    bboxes2_area = (bboxes2[:, [2]] - bboxes2[:, [0]] + 1) * (
        bboxes2[:, [3]] - bboxes2[:, [1]] + 1
    )

    # Calculate the union as the sum of areas minus intersection
    union = (bboxes1_area + bboxes2_area.T) - intersection

    # We start we an empty array of zeros.
    iou = np.zeros((bboxes1.shape[0], bboxes2.shape[0]))

    # Only divide where the intersection is > 0
    np.divide(intersection, union, out=iou, where=intersection > 0.0)
    return iou
