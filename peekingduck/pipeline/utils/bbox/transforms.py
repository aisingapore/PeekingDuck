# Copyright 2022 AI Singapore
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

"""Utility functions which convert bbox coordinates from one format to
another.
"""

import numpy as np
import torch


def tlwh2xyah(inputs: np.ndarray) -> np.ndarray:
    """Converts from [t, l, w, h] to [x, y, a, h] format.

    (t, l) is the coordinates of the top left corner, w is the width, and h is
    the height. (x, y) is the coordinates of the object center, `a` is the
    aspect ratio, and `h` is the height. Aspect ratio is `width / height`.

    [x, y, a, h] is calculated as:
    x = (t + w) / 2
    y = (l + h) / 2
    a = w / h
    h = h

    Example:
        >>> a = tlwh2xyah(inputs=np.array([1.0, 2.0, 30.0, 40.0]))
        >>> a
        array([16.0, 22.0, 0.75, 40.0])

    Args:
        inputs (np.ndarray): Input bounding box (1-d array) with the format
            `(top left x, top left y, width, height)`.

    Returns:
        (np.ndarray): Bounding box with the format `(center x, center y, aspect
        ratio,height)`.
    """
    outputs = np.asarray(inputs).copy()
    outputs[:2] += outputs[2:] / 2
    outputs[2] /= outputs[3]
    return outputs


def tlwh2xyxyn(inputs: np.ndarray, height: int, width: int) -> np.ndarray:
    """Converts from [t, l, w, h] to normalized [x1, y1, x2, y2] format.
    Normalized coordinates are w.r.t. original image size.

    (t, l) is the coordinates of the top left corner, w is the width, and h is
    the height. (x1, y1) and (x2, y2) are the normalized coordinates of top
    left and bottom right, respectively.

    [x1, y1, x2, y2] is calculated as:
    x1 = t / width
    y1 = l / height
    x2 = (t + w) / width
    y2 = (l + h) / height

    Example:
        >>> a = tlwh2xyxyn(inputs=np.array([[1.0, 2.0, 30.0, 40.0]]), height=100, width=200)
        >>> a
        array([[0.005, 0.02, 0.155, 0.42]])

    Args:
        inputs (np.ndarray): Input bounding boxes (2-d array) each with the
            format `(top left x, top left y, width, height)`.
        height (int): Height of the image frame.
        height (int): Width of the image frame.

    Returns:
        (np.ndarray): Bounding boxes with the format `normalized (top left x,
        top left y, bottom right x, bottom right y)`.
    """
    outputs = np.empty_like(inputs)
    outputs[:, 0] = inputs[:, 0] / width
    outputs[:, 1] = inputs[:, 1] / height
    outputs[:, 2] = (inputs[:, 0] + inputs[:, 2]) / width
    outputs[:, 3] = (inputs[:, 1] + inputs[:, 3]) / height
    return outputs


def xywh2xyxy(inputs: torch.Tensor) -> torch.Tensor:
    """Converts from [x, y, w, h] to [x1, y1, x2, y2] format.

    (x, y) is the object center, w is the width, and h is the height. (x1, y1)
    is the top left corner and (x2, y2) is the bottom right corner.

    [x1, y1, x2, y2] is calculated as:
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + w / 2

    Example:
        >>> a = xywh2xyxy(inputs=torch.Tensor([[100, 200, 30, 40]]))
        >>> a
        tensor([[ 85.0, 180.0, 115.0, 220.0]])

    Args:
        inputs (torch.Tensor): Input bounding boxes (2-d array) each with the
            format `(center x, center y, width, height)`.

    Returns:
        (torch.Tensor): Bounding boxes with the format `(top left x, top left y,
        bottom right x, bottom right y)`.
    """
    outputs = torch.empty_like(inputs)
    outputs[:, 0] = inputs[:, 0] - inputs[:, 2] / 2
    outputs[:, 1] = inputs[:, 1] - inputs[:, 3] / 2
    outputs[:, 2] = inputs[:, 0] + inputs[:, 2] / 2
    outputs[:, 3] = inputs[:, 1] + inputs[:, 3] / 2

    return outputs


def xyxy2tlwh(inputs: np.ndarray) -> np.ndarray:
    """Converts from [x1, y1, x2, y2] to [t, l, w, h].

    (x1, y1) and (x2, y2) are the coordinates of top left and bottom right,
    respectively. (t, l) is the coordinates of the top left corner, w is the
    width, and h is the height.

    [t, l, w, h] is calculated as:
    t = x1
    l = y1
    w = x2 - x1
    h = y2 - y1

    Example:
        >>> a = xyxy2tlwh(inputs=np.array([1.0, 2.0, 30.0, 40.0]))
        >>> a
        array([1, 2, 29, 38])

    Args:
        inputs (np.ndarray): Input bounding box (1-d array) each with the
            format `(top left x, top left y, bottom right x, bottom right y)`.

    Returns:
        (np.ndarray): Bounding box with the format `(top left x, top left y,
        width, height)`.
    """
    outputs = np.asarray(inputs).copy()
    outputs[2:] -= outputs[:2]
    return outputs


def xyxy2xyxyn(inputs: np.ndarray, height: float, width: float) -> np.ndarray:
    """Converts from [x1, y1, x2, y2] to normalized [x1, y1, x2, y2].
    Normalized coordinates are w.r.t. original image size.

    (x1, y1) is the top left corner and (x2, y2) is the bottom right corner.

    Normalized [x1, y1, x2, y2] is calculated as:
    Normalized x1 = x1 / width
    Normalized y1 = y1 / height
    Normalized x2 = x2 / width
    Normalized y2 = y2 / height

    Example:
        >>> a = xyxy2xyxyn(inputs=np.array([[1.0, 2.0, 30.0, 40.0]]), height=100, width=200)
        >>> a
        array([[0.005, 0.02, 0.15, 0.4]])

    Args:
        inputs (np.ndarray): Input bounding boxes (2-d array) each with the
            format `(top left x, top left y, bottom right x, bottom right y)`.

    Returns:
        (np.ndarray): Bounding boxes with the format `normalized (top left x,
        top left y, bottom right x, bottom right y)`.
    """
    outputs = np.empty_like(inputs)
    outputs[:, [0, 2]] = inputs[:, [0, 2]] / width
    outputs[:, [1, 3]] = inputs[:, [1, 3]] / height

    return outputs


def xyxyn2tlwh(inputs: np.ndarray, height: float, width: float) -> np.ndarray:
    """Converts from normalized [x1, y1, x2, y2] to [t, l, w, h] format.
    Normalized coordinates are w.r.t. original image size.

    (t, l) is the coordinates of the top left corner, w is the width, and h is
    the height. (x1, y1) and (x2, y2) are the normalized coordinates of top
    left and bottom right, respectively.

    [t, l, w, h] is calculated as:
    t = x1 * width
    l = y1 * height
    w = (x2 - x1) * width
    h = (y2 - y1) * height

    Example:
        >>> a = xyxyn2tlwh(inputs=np.array([[0.0, 0.02, 0.3, 0.4]]), height=100, width=200)
        >>> a
        array([[ 0.0  2.0 60.0 38.0]])

    Args:
        inputs (np.ndarray): Input bounding boxes (2-d array) each with the
            format `normalized (top left x, top left y, bottom right x, bottom
            right y)`.
        height (int): Height of the image frame.
        height (int): Width of the image frame.

    Returns:
        (np.ndarray): Bounding boxes with the format `(top left x, top left y,
        width, height)`.
    """
    outputs = np.empty_like(inputs)
    outputs[:, 0] = inputs[:, 0] * width
    outputs[:, 1] = inputs[:, 1] * height
    outputs[:, 2] = (inputs[:, 2] - inputs[:, 0]) * width
    outputs[:, 3] = (inputs[:, 3] - inputs[:, 1]) * height

    return outputs
