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

Modifications:
1. Allow bbox inputs to be either numpy array or torch tensor. Created a type BboxType.
2. Added clone() function to avoid inplace mutation.
3. Added cast_int_to_float() function to convert int to float to avoid output become all 0.
4. Added new transforms voc2yolo, yolo2voc, albu2yolo, yolo2albu, xyxyn2xyxy, xyxy2xywh
"""


from typing import Union

import numpy as np
import torch

BboxType = Union[np.ndarray, torch.Tensor]


def cast_int_to_float(inputs: BboxType) -> BboxType:
    """Converts int to float.

    Args:
        inputs (BboxType): Input bounding box.

    Returns:
        (BboxType): Cast input bounding box to float type.
    """
    if isinstance(inputs, torch.Tensor):
        return inputs.float()
    return inputs.astype(np.float32)


def clone(inputs: BboxType) -> BboxType:
    """Clones bounding box to avoid inplace mutation.

    Note:
        copy.deepcopy() does not work for autograd tensors.

    Args:
        inputs (BboxType): Input bounding box.

    Returns:
        (BboxType): Clone of input bounding box.
    """
    if isinstance(inputs, torch.Tensor):
        return inputs.clone()
    return inputs.copy()


def tlwh2xyah(inputs: BboxType) -> BboxType:
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
        inputs (BboxType): Input bounding box of shape (..., 4) with the format
            `(top left x, top left y, width, height)`.

    Returns:
        outputs (BboxType): Bounding box with the format `(center x, center y, aspect
        ratio,height)`.
    """
    outputs = clone(inputs)
    outputs = cast_int_to_float(outputs)

    outputs[..., :2] += outputs[..., 2:] / 2
    outputs[..., 2] /= outputs[..., 3]
    return outputs


def yolo2albu(inputs: BboxType) -> BboxType:
    """Converts from normalized [x, y, w, h] to [x1, y1, x2, y2] normalized format.

    (x, y): the normalized coordinates of the center of the bounding box;
    (w, h): the normalized width and height of the bounding box.
    (x1, y1): the normalized coordinates of the top left corner of the bounding box;
    (x2, y2): the normalized coordinates of the bottom right corner of the bounding box.

    [x1, y1, x2, y2] is calculated as:
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x1 + w
    y2 = y1 + h

    Args:
        inputs (BboxType): Input bounding boxes of shape (..., 4) with the format
            `(center x, center y, width, height)` normalized by image width and
            height.

    Returns:
        outputs (BboxType): Bounding boxes of shape (..., 4) with the format `(top
            left x, top left y, bottom right x, bottom right y)` normalized by
            image width and height.
    """
    outputs = clone(inputs)
    outputs = cast_int_to_float(outputs)

    outputs[..., 0] -= outputs[..., 2] / 2
    outputs[..., 1] -= outputs[..., 3] / 2
    outputs[..., 2] += outputs[..., 0]
    outputs[..., 3] += outputs[..., 1]

    return outputs


def albu2yolo(inputs: BboxType) -> BboxType:
    """Converts from [x1, y1, x2, y2] normalized format to normalized [x, y, w, h].

    (x1, y1): the normalized coordinates of the top left corner of the bounding box;
    (x2, y2): the normalized coordinates of the bottom right corner of the bounding box.
    (x, y): the normalized coordinates of the center of the bounding box;
    (w, h): the normalized width and height of the bounding box.

    [x, y, w, h] is calculated as:
    w = x2 - x1
    h = y2 - y1
    x = x1 + w / 2
    y = y1 + h / 2

    Args:
        inputs (BboxType): Input bounding boxes of shape (..., 4) with the format
            `(top left x, top left y, bottom right x, bottom right y)` normalized by
            image width and height.

    Returns:
        outputs (BboxType): Bounding boxes of shape (..., 4) with the format `(center
            x, center y, width, height)` normalized by image width and height.
    """
    outputs = clone(inputs)
    outputs = cast_int_to_float(outputs)

    outputs[..., 2] -= outputs[..., 0]
    outputs[..., 3] -= outputs[..., 1]
    outputs[..., 0] += outputs[..., 2] / 2
    outputs[..., 1] += outputs[..., 3] / 2

    return outputs


def voc2yolo(inputs: BboxType, height: float, width: float) -> BboxType:
    """Converts from [x1, y1, x2, y2] to normalized [x, y, w, h] format.

    (x1, y1): the coordinates of the top left corner of the bounding box;
    (x2, y2): the coordinates of the bottom right corner of the bounding box.
    (x, y): the coordinates of the center of the bounding box;
    (w, h): the width and height of the bounding box.

    [x, y, w, h] is calculated as:
    x = (x1 + x2) / 2 / width
    y = (y1 + y2) / 2 / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height

    Args:
        inputs (BboxType): Input bounding boxes of shape (..., 4) with the format
            `(top left x, top left y, bottom right x, bottom right y)`.
        height (float): Height of the image frame.
        width (float): Width of the image frame.

    Returns:
        outputs (BboxType): Bounding boxes of shape (..., 4) with the format `(center x,
            center y, width, height)` normalized by image width and height.
    """
    outputs = clone(inputs)
    outputs = cast_int_to_float(outputs)

    outputs[..., [0, 2]] /= width
    outputs[..., [1, 3]] /= height

    outputs[..., 2] -= outputs[..., 0]
    outputs[..., 3] -= outputs[..., 1]

    outputs[..., 0] += outputs[..., 2] / 2
    outputs[..., 1] += outputs[..., 3] / 2

    return outputs


def yolo2voc(inputs: BboxType, height: float, width: float) -> BboxType:
    """Converts from normalized [x, y, w, h] to [x1, y1, x2, y2] format.

    (x, y): the coordinates of the center of the bounding box;
    (w, h): the width and height of the bounding box.
    (x1, y1): the coordinates of the top left corner of the bounding box;
    (x2, y2): the coordinates of the bottom right corner of the bounding box.

    [x1, y1, x2, y2] is calculated as:
    x1 = x * width - w * width / 2
    y1 = y * height - h * height / 2
    x2 = x * width + w * width / 2
    y2 = y * height + h * height / 2

    Args:
        inputs (BboxType): Input bounding boxes of shape (..., 4) with the format
            `(center x, center y, width, height)` normalized by image width and height.
        height (float): Height of the image frame.
        width (float): Width of the image frame.

    Returns:
        outputs (BboxType): Bounding boxes of shape (..., 4) with the format `(top
            left x, top left y, bottom right x, bottom right y)`.
    """
    outputs = clone(inputs)
    outputs = cast_int_to_float(outputs)

    outputs[..., [0, 2]] *= width
    outputs[..., [1, 3]] *= height

    outputs[..., 0] -= outputs[..., 2] / 2
    outputs[..., 1] -= outputs[..., 3] / 2
    outputs[..., 2] += outputs[..., 0]
    outputs[..., 3] += outputs[..., 1]

    return outputs


def xyxy2xywh(inputs: BboxType) -> BboxType:
    """Converts from [x1, y1, x2, y2] to [x, y, w, h] format.

    (x, y) is the object center, w is the width, and h is the height. (x1, y1)
    is the top left corner and (x2, y2) is the bottom right corner.

    [x, y, w, h] is calculated as:
    w = x2 - x1
    h = y2 - y1
    x = x1 + w / 2
    y = y1 + h / 2

    Args:
        inputs (BboxType): Input bounding boxes of shape (..., 4) with the format
            `(top left x, top left y, bottom right x, bottom right y)`.

    Returns:
        outputs (BboxType): Bounding boxes of shape (..., 4) with the format `(center x,
            center y, width, height)`.
    """
    outputs = clone(inputs)
    outputs = cast_int_to_float(outputs)

    outputs[..., 2] = inputs[..., 2] - inputs[..., 0]
    outputs[..., 3] = inputs[..., 3] - inputs[..., 1]
    outputs[..., 0] += outputs[..., 2] / 2
    outputs[..., 1] += outputs[..., 3] / 2

    return outputs


def xywh2xyxy(inputs: BboxType) -> BboxType:
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
        inputs (BboxType): Input bounding boxes of shape (..., 4) each with the
            format `(center x, center y, width, height)`.

    Returns:
        outputs (BboxType): Bounding boxes with the format `(top left x, top left y,
            bottom right x, bottom right y)`.
    """
    outputs = clone(inputs)
    outputs = cast_int_to_float(outputs)

    outputs[..., 0] = inputs[..., 0] - inputs[..., 2] / 2
    outputs[..., 1] = inputs[..., 1] - inputs[..., 3] / 2
    outputs[..., 2] = inputs[..., 0] + inputs[..., 2] / 2
    outputs[..., 3] = inputs[..., 1] + inputs[..., 3] / 2

    return outputs


def tlwh2xyxy(inputs: BboxType) -> BboxType:
    """Converts from [t, l, w, h] to [x1, y1, x2, y2] format.

    (t, l) is the coordinates of the top left corner, w is the width, and h is
    the height. (x1, y1) is the top left corner and (x2, y2) is the bottom
    right corner.

    [x1, y1, x2, y2] is calculated as:
    x1 = t
    y1 = l
    x2 = t + w
    y2 = l + h

    Example:
        >>> a = tlwh2xyxy(inputs=torch.Tensor([[1.0, 2.0, 30.0, 40.0]]))
        >>> a
        tensor([[ 1.0,  2.0, 31.0, 42.0]])

    Args:
        inputs (BboxType): Input bounding boxes of shape (..., 4) each with the
            format `(top left x, top left y, width, height)`.

    Returns:
        outputs (BboxType): Bounding boxes with the format `(top left x, top left y,
            bottom right x, bottom right y)`.
    """
    outputs = clone(inputs)
    outputs = cast_int_to_float(outputs)

    outputs[..., 2:] += outputs[..., :2]
    return outputs


def xyxy2tlwh(inputs: BboxType) -> BboxType:
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
        inputs (BboxType): Input bounding box of shape (..., 4) each with the
            format `(top left x, top left y, bottom right x, bottom right y)`.

    Returns:
        outputs (BboxType): Bounding box with the format `(top left x, top left y,
            width, height)`.
    """
    outputs = clone(inputs)
    outputs = cast_int_to_float(outputs)

    outputs[..., 2:] -= outputs[..., :2]

    return outputs


def xyxy2xyxyn(inputs: BboxType, height: float, width: float) -> BboxType:
    """Converts from [x1, y1, x2, y2] to normalized [x1, y1, x2, y2].

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
        inputs (BboxType): Input bounding boxes of shape (..., 4) each with the
            format `(top left x, top left y, bottom right x, bottom right y)`.

    Returns:
        outputs (BboxType): Bounding boxes with the format `normalized (top left x,
            top left y, bottom right x, bottom right y)`.
    """
    outputs = clone(inputs)
    outputs = cast_int_to_float(outputs)

    outputs[..., [0, 2]] = inputs[..., [0, 2]] / width
    outputs[..., [1, 3]] = inputs[..., [1, 3]] / height

    return outputs


def xyxyn2xyxy(inputs: BboxType, height: float, width: float) -> BboxType:
    """Converts from normalized [x1, y1, x2, y2] to [x1, y1, x2, y2].

    (x1, y1) is the top left corner and (x2, y2) is the bottom right corner.

    [x1, y1, x2, y2] is calculated as:
    x1 = Normalized x1 * width
    y1 = Normalized y1 * height
    x2 = Normalized x2 * width
    y2 = Normalized y2 * height

    Example:
        >>> a = xyxyn2xyxy(inputs=np.array([[0.005, 0.02, 0.15, 0.4]]), height=100, width=200)
        >>> a
        array([[1.0, 2.0, 30.0, 40.0]])

    Args:
        inputs (BboxType): Input bounding boxes of shape (..., 4) each with the
            format `normalized (top left x, top left y, bottom right x, bottom
            right y)`.

    Returns:
        outputs (BboxType): Bounding boxes with the format `(top left x, top left y,
            bottom right x, bottom right y)`.
    """
    outputs = clone(inputs)
    outputs = cast_int_to_float(outputs)

    outputs[..., [0, 2]] = outputs[..., [0, 2]] * width
    outputs[..., [1, 3]] = outputs[..., [1, 3]] * height

    return outputs


def xyxyn2tlwh(inputs: BboxType, height: float, width: float) -> BboxType:
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
        inputs (BboxType): Input bounding boxes with shape (..., 4) each with the
            format `normalized (top left x, top left y, bottom right x, bottom
            right y)`.
        height (int): Height of the image frame.
        height (int): Width of the image frame.

    Returns:
        outputs (BboxType): Bounding boxes with the format `(top left x, top left y,
            width, height)`.
    """
    outputs = clone(inputs)
    outputs = cast_int_to_float(outputs)

    outputs[..., 0] = inputs[..., 0] * width
    outputs[..., 1] = inputs[..., 1] * height
    outputs[..., 2] = (inputs[..., 2] - inputs[..., 0]) * width
    outputs[..., 3] = (inputs[..., 3] - inputs[..., 1]) * height

    return outputs


def tlwh2xyxyn(inputs: BboxType, height: float, width: float) -> BboxType:
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
        inputs (BboxType): Input bounding boxes of variable shape (..., 4) with
            format `(top left x, top left y, width, height)`.
        height (int): Height of the image frame.
        height (int): Width of the image frame.

    Returns:
        outputs (BboxType): Bounding boxes with the format `normalized (top left x,
            top left y, bottom right x, bottom right y)`.
    """
    outputs = clone(inputs)
    outputs = cast_int_to_float(outputs)

    outputs[..., 0] = inputs[..., 0] / width
    outputs[..., 1] = inputs[..., 1] / height
    outputs[..., 2] = (inputs[..., 0] + inputs[..., 2]) / width
    outputs[..., 3] = (inputs[..., 1] + inputs[..., 3]) / height

    return outputs
