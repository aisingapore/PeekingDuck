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

"""Utility functions used by tracking-by-detection trackers."""

import numpy as np


def iou_candidates(bbox: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Computes Intersection-over-Union of `bbox` and each of the candidate
    bbox in `candidates`.

    Args:
        bbox (np.ndarray): A bounding box in format `(top left x, top left y,
            width, height)`.
        candidates (np.ndarray): A matrix of candidate bounding boxes
            (one per row) in the same format as `bbox`.
    Returns:
        np.ndarray: The IoU in [0, 1] between the `bbox` and each candidate. A
            higher score means a larger fraction of the `bbox` is occluded by
            the candidate.
    """
    bbox_tl = bbox[:2]
    bbox_br = bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    top_left = np.maximum(bbox_tl, candidates_tl)
    bottom_right = np.minimum(bbox_br, candidates_br)
    width_height = np.maximum(0.0, bottom_right - top_left)

    area_intersection = width_height[:, 0] * width_height[:, 1]
    area_bbox = bbox[2] * bbox[3]
    area_candidates = candidates[:, 2] * candidates[:, 3]

    return area_intersection / (area_bbox + area_candidates - area_intersection)


def xyxyn2tlwh(inputs: np.ndarray, height: int, width: int) -> np.ndarray:
    """Converts bounding boxes format from (x1, y2, x2, y2) to (t, l, w, h).
    (x1, y1) is the normalized coordinates of the top-left corner, (x2, y2) is
    the normalized coordinates of the bottom-right corner. (t, l) is the
    original coordinates of the top-left corner, (w, h) is the original width
    and height of the bounding box.

    Args:
        inputs (np.ndarray): Bounding box coordinates with (x1, y1, x2, y2)
            format.
        height (int): Original height of bounding box.
        width (int): Original width of bounding box.

    Returns:
        (np.ndarray): Converted bounding box coordinates with (t, l, w, h)
            format.
    """
    outputs = np.empty_like(inputs)
    outputs[:, 0] = inputs[:, 0] * width  # Bottom left x
    outputs[:, 1] = inputs[:, 1] * height  # Bottom left y
    outputs[:, 2] = (inputs[:, 2] - inputs[:, 0]) * width  # Top right x
    outputs[:, 3] = (inputs[:, 3] - inputs[:, 1]) * height  # Top right y
    return outputs
