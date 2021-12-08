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
Functions for bounding box matching
"""

from typing import Any, List, Tuple, Union
import lap
import numpy as np
from scipy.spatial.distance import cdist
from cython_bbox import bbox_overlaps as bbox_ious
from peekingduck.pipeline.nodes.model.jde_mot.jde_files.utils import kalman_filter
from peekingduck.pipeline.nodes.model.jde_mot.jde_files.utils.kalman_filter import (
    KalmanFilter,
)

# pylint: disable=invalid-name, no-name-in-module, unused-argument, redefined-outer-name, too-many-arguments


def linear_assignment(
    cost_matrix: np.ndarray, thresh: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Function for Hungarian Algorithm association.

    Args:
        cost_matrix (np.ndarray): Cost matrix of current and previous matrices.
        thresh (float): Threshold value

    Returns:
        (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains arrays of matched and unmatched tracks.
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    matches, unmatched_a, unmatched_b = [], [], []
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)

    return matches, unmatched_a, unmatched_b


def ious(
    atlbrs: Union[List[Any], np.ndarray], btlbrs: Union[List[Any], np.ndarray]
) -> np.ndarray:
    """Compute cost based on IoU.

    Args:
        atlbrs (list[tlbr] | np.ndarray): List of STracks.
        btlbrs (list[tlbr] | np.ndarray): List of STracks.

    Returns:
        np.ndarray: Array of intersection over union.
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float),
    )

    return ious


def iou_distance(atracks: List[Any], btracks: List[Any]) -> np.ndarray:
    """Compute cost based on IoU.

    Args:
        atracks (list[STrack]): List of STracks.
        btracks (list[STrack]): List of STracks.

    Returns:
        np.ndarray: Cost matrix of distance between intersection over
            union of bounding boxes.
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(
    tracks: List[Any], detections: List[Any], metric: str = "cosine"
) -> np.ndarray:
    """Calculate distance joint STracks and detections.

    Args:
        tracks (List[STrack]): STracks
        detections (List[BaseTrack]): Predictions of model
        metric (str, optional): Type of distance metric to use. Defaults to "cosine".

    Returns:
        np.ndarray: Cost matrix of distance.
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(
        0.0, cdist(track_features, det_features)
    )  # Nomalized features

    return cost_matrix


def fuse_motion(
    kf: KalmanFilter,
    cost_matrix: np.ndarray,
    tracks: List[Any],
    detections: List[Any],
    only_position: bool = False,
    lambda_: float = 0.98,
) -> np.ndarray:
    """Fuse various motions of detections and STracks.

    Args:
        kf (KalmanFilter): Kalman filter module.
        cost_matrix (np.ndarray): Cost matrix of embedded distances.
        tracks (List[Any]): STracks.
        detections (List[Any]): Predictions of model.
        only_position (bool, optional): Gating dimension. Defaults to False.
        lambda_ (float, optional): Threshold parameter. Defaults to 0.98.

    Returns:
        np.ndarray: Distances of fused motion.
    """
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric="maha"
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix
