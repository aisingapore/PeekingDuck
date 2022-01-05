# Copyright 2021 AI Singapore
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

    top_left = np.c_[
        np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
        np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis],
    ]
    bottom_right = np.c_[
        np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
        np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis],
    ]
    width_height = np.maximum(0.0, bottom_right - top_left)

    area_intersection = width_height.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)

    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_tlwh(bbox_1: np.ndarray, bbox_2: np.ndarray) -> float:
    """Calculates the Intersection-over-Union (IoU) of two bounding boxes. Each
    bounding box as the format (t, l, w, h) where (t, l) is the top-left
    corner, w is the width, and h is the height.

    Args:
        bbox_1 (np.ndarray): The first bounding box.
        bbox_2 (np.ndarray): The other bounding box.

    Returns:
        (float): IoU of bbox1, bbox2.
    """
    bbox_1 = bbox_1.astype(float)
    bbox_2 = bbox_2.astype(float)
    overlap_x1 = np.maximum(bbox_1[0], bbox_2[0])
    overlap_y1 = np.maximum(bbox_1[1], bbox_2[1])
    overlap_x2 = np.minimum(bbox_1[0] + bbox_1[2], bbox_2[0] + bbox_2[2])
    overlap_y2 = np.minimum(bbox_1[1] + bbox_1[3], bbox_2[1] + bbox_2[3])

    overlap_w = max(0.0, overlap_x2 - overlap_x1)
    overlap_h = max(0.0, overlap_y2 - overlap_y1)
    if overlap_w <= 0 or overlap_h <= 0:
        return 0.0

    area_intersection = overlap_w * overlap_h
    area_bbox_1 = bbox_1[2:].prod()
    area_bbox_2 = bbox_2[2:].prod()

    return area_intersection / (area_bbox_1 + area_bbox_2 - area_intersection)


def xyxyn2tlwh(inputs: np.ndarray, height: int, width: int) -> np.ndarray:
    """Converts bounding boxes format from (x1, y2, x2, y2) to (t, l, w, h).
    (x1, y1) is the normalised coordinates of the top-left corner, (x2, y2) is
    the normalised coordinates of the bottom-right corner. (t, l) is the
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
