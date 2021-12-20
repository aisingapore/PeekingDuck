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

"""Functions for bounding box matching.

Modifications include:
- Pure python replacement of cython_bbox
- Removed checking for List[np.ndarray] types in iou_distance()
- Set return_cost=False and use list comprehension in linear_assignment()
- Removed only_position argument in fuse_motion as only False value is used.
"""

from typing import List, Tuple

import lap
import numpy as np
from scipy.spatial.distance import cdist

from peekingduck.pipeline.nodes.model.jdev1.jde_files.kalman_filter import (
    KalmanFilter,
    chi2inv95,
)
from peekingduck.pipeline.nodes.model.jdev1.jde_files.track import STrack


def bbox_ious(bboxes_1: np.ndarray, bboxes_2: np.ndarray) -> np.ndarray:
    """Calculates the Intersection-over-Union (IoU) between bounding boxes.
    Bounding boxes have the format (x1, y1, x2, y2), where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. The algorithm is
    adapted from simple-faster-rcnn-pytorch with modifications such as adding
    1 to area calculations to match the equations used in cython_bbox.

    Args:
        bboxes_1 (np.ndarray): An array with shape (N, 4) where N is the number
            of bounding boxes.
        bboxes_2 (np.ndarray): An array similar to `bboxes_2` with shape (K, 4)
            where K is the number of bounding boxes.

    Returns:
        (np.ndarray): An array with shape (N, K). The element at index (n, k)
        contains the IoU between the n-th bounding box in `bboxes_1` and the
        k-th bounding box in `bboxes_2`.

    Reference:
        simple-faster-rcnn-pytorch
        https://github.com/chenyuntc/simple-faster-rcnn-pytorch

        cython_bbox:
        https://github.com/samson-wang/cython_bbox
    """
    # top left
    intersect_tl = np.maximum(bboxes_1[:, np.newaxis, :2], bboxes_2[:, :2])
    # bottom right
    intersect_br = np.minimum(bboxes_1[:, np.newaxis, 2:], bboxes_2[:, 2:]) + 1

    intersect_area = np.prod(intersect_br - intersect_tl, axis=2) * (
        intersect_tl < intersect_br
    ).all(axis=2)
    area_1 = np.prod(bboxes_1[:, 2:] - bboxes_1[:, :2] + 1, axis=1)
    area_2 = np.prod(bboxes_2[:, 2:] - bboxes_2[:, :2] + 1, axis=1)
    iou_values = intersect_area / (area_1[:, np.newaxis] + area_2 - intersect_area)

    return iou_values


def embedding_distance(
    tracks: List[STrack], detections: List[STrack], metric: str = "euclidean"
) -> np.ndarray:
    """Computes cost based on features between `tracks` and `detections`.

    Args:
        tracks (List[STrack]): List of STracks.
        detections (List[STrack]): List of STracks that are model predictions.
        metric (str): The metric to be used with
            `scipy.spatial.distance.cdist()`. Defaults to "euclidean".

    Returns:
        np.ndarray: Cost matrix of distance.
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    # Normalised features
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))

    return cost_matrix


def fuse_motion(  # pylint: disable=too-many-arguments
    kalman_filter: KalmanFilter,
    cost_matrix: np.ndarray,
    tracks: List[STrack],
    detections: List[STrack],
    coeff: float = 0.98,
) -> np.ndarray:
    """Computes the cost matrix using the pair-wise motion affinity matrix and
    appearance affinity matrix.

    Args:
        kalman_filter (KalmanFilter): Kalman filter for state estimation.
        cost_matrix (np.ndarray): Cost matrix filled with values from the
            appearance affinity matrix.
        tracks (List[STrack]): List of STracks.
        detections (List[STrack]): List of STracks that are model predictions.
        coeff (float): Weighting parameter used in computing the final cost
            matrix, corresponds to `lambda` in the arxiv article.

    Returns:
        (np.ndarray): Cost matrix used by Hungarian algorithm to solve the
        linear assignment problem.
    """
    if cost_matrix.size == 0:
        return cost_matrix
    gating_threshold = chi2inv95[4]
    measurements = np.asarray([det.xyah for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kalman_filter.gating_distance(
            track.mean, track.covariance, measurements
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = coeff * cost_matrix[row] + (1 - coeff) * gating_distance
    return cost_matrix


def iou_distance(tracks_1: List[STrack], tracks_2: List[STrack]) -> np.ndarray:
    """Computes cost based on Intersection-over-Union (IoU).

    Args:
        tracks_1 (List[STrack]): List of STracks.
        tracks_2 (List[STrack]): List of STracks.

    Returns:
        (np.ndarray): Cost matrix of distance between IoU of bounding boxes.
    """
    xyxys_1 = [track.xyxy for track in tracks_1]
    xyxys_2 = [track.xyxy for track in tracks_2]
    cost_matrix = 1 - ious(xyxys_1, xyxys_2)

    return cost_matrix


def ious(xyxys_1: List[np.ndarray], xyxys_2: List[np.ndarray]) -> np.ndarray:
    """Computes a matrix Intersection-over-Union (IoU) values between 2 list
    of bounding boxes with (x1, y1, x2, y2) format where (x1, y1) is the top
    left and (x2, y2) is the bottom right.

    Args:
        xyxys_1 (List[np.ndarray]): List of STracks.
        xyxys_2 (List[np.ndarray]): List of STracks.

    Returns:
        np.ndarray: Matrix of IoU values.
    """
    iou_values = np.zeros((len(xyxys_1), len(xyxys_2)), dtype=np.float)
    if iou_values.size == 0:
        return iou_values

    return bbox_ious(
        np.ascontiguousarray(xyxys_1, dtype=np.float),
        np.ascontiguousarray(xyxys_2, dtype=np.float),
    )


def linear_assignment(
    cost_matrix: np.ndarray, threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uses Hungarian Algorithm to associate detections to tracks.

    Args:
        cost_matrix (np.ndarray): Cost matrix which is a weighted sum of the
            pair-wise motion affinity matrix and appearance affinity matrix.
        threshold (float): An upper limit for a cost of a single assignment.

    Returns:
        (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains arrays of matched and unmatched tracks.
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0], dtype=int),
            np.arange(cost_matrix.shape[1], dtype=int),
        )
    x_assignment, y_assignment = lap.lapjv(
        cost_matrix, extend_cost=True, cost_limit=threshold, return_cost=False
    )
    matches = np.asarray(
        [[row, col] for row, col in enumerate(x_assignment) if col >= 0]
    )
    unmatched_1 = np.where(x_assignment < 0)[0]
    unmatched_2 = np.where(y_assignment < 0)[0]
    return matches, unmatched_1, unmatched_2
