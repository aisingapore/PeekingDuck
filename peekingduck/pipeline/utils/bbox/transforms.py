"""Utility functions which convert bbox coordinates from one format to
another.
"""

import numpy as np
import torch


def tlwh2xyah(inputs: np.ndarray) -> np.ndarray:
    """Converts bounding box to format `(center x, center y, aspect ratio,
    height)`, where the aspect ratio is `width / height`.

    Args:
        tlwh (np.ndarray): Input bounding box with format `(top left x,
            top left y, width, height)`.
    Returns:
        (np.ndarray): Bounding box with (x, y, a, h) format.
    """
    outputs = np.asarray(inputs).copy()
    outputs[:2] += outputs[2:] / 2
    outputs[2] /= outputs[3]
    return outputs


def tlwh2xyxyn(inputs: np.ndarray, height: int, width: int) -> np.ndarray:
    """Converts from [t, l, w, h] to [x1, y1, x2, y2] format.

    (x1, y1) and (x2, y2) are coordinates of top left and bottom right
    respectively. (t, l) is the coordinates of the top left corner, w is the
    width, and h is the height.
    """
    outputs = np.empty_like(inputs)
    outputs[:, 0] = inputs[:, 0] / width
    outputs[:, 1] = inputs[:, 1] / height
    outputs[:, 2] = (inputs[:, 0] + inputs[:, 2]) / width
    outputs[:, 3] = (inputs[:, 1] + inputs[:, 3]) / height
    return outputs


def xywh2xyxy(inputs: torch.Tensor) -> torch.Tensor:
    """Converts from [x, y, w, h] to [x1, y1, x2, y2] format.

    (x, y) is the object center. (x1, y1) is the top left corner and (x2, y2)
    is the bottom right corner.
    """
    outputs = torch.empty_like(inputs)
    outputs[:, 0] = inputs[:, 0] - inputs[:, 2] / 2
    outputs[:, 1] = inputs[:, 1] - inputs[:, 3] / 2
    outputs[:, 2] = inputs[:, 0] + inputs[:, 2] / 2
    outputs[:, 3] = inputs[:, 1] + inputs[:, 3] / 2

    return outputs


def xyxy2tlwh(inputs: np.ndarray) -> np.ndarray:
    """Converts bounding box to format `(top left x, top left y, width,
    height)`.

    Args:
        inputs (np.ndarray): Input bounding box with format (x1, y1, x2, y2)
            where (x1, y1) is top left, (x2, y2) is bottom right.

    Returns:
        (np.ndarray): Bounding box with `(top left x, top left y, width,
            height)` format.
    """
    outputs = np.asarray(inputs).copy()
    outputs[2:] -= outputs[:2]
    return outputs


def xyxy2xyxyn(inputs: np.ndarray, height: float, width: float) -> np.ndarray:
    """Converts from [x1, y1, x2, y2] to normalized [x1, y1, x2, y2].

    (x1, y1) is the top left corner and (x2, y2) is the bottom right corner.
    Normalized coordinates are w.r.t. original image size.
    """
    outputs = np.empty_like(inputs)
    outputs[:, [0, 2]] = inputs[:, [0, 2]] / width
    outputs[:, [1, 3]] = inputs[:, [1, 3]] / height

    return outputs
