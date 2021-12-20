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
Functions for bounding box matching.
"""

from typing import Any, List, Tuple
import lap
import numpy as np
from scipy.spatial.distance import cdist
from peekingduck.pipeline.nodes.model.jdev1.jde_files.utils.utils import (
    bbox_overlap,
)
from peekingduck.pipeline.nodes.model.jdev1.jde_files.utils.kalman_filter import (
    chi2inv95,
    KalmanFilter,
)


def linear_assignment(
    cost_matrix: np.ndarray, thresh: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Function for Hungarian Algorithm association.

    Args:
        cost_matrix (np.ndarray): Cost matrix of current and previous matrices.
        thresh (float): Threshold value.

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
    _, x_array, y_array = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for x_index, x_iterable in enumerate(x_array):
        if x_iterable >= 0:
            matches.append([x_index, x_iterable])
    unmatched_a = np.where(x_array < 0)[0]
    unmatched_b = np.where(y_array < 0)[0]
    matches = np.asarray(matches)

    return matches, unmatched_a, unmatched_b


def ious(atlbrs: List[Any], btlbrs: List[Any]) -> np.ndarray:
    """Compute cost based on Intersection over Union.

    Args:
        atlbrs (List[Any]): List of STracks.
        btlbrs (List[Any]): List of STracks.

    Returns:
        np.ndarray: Array of intersection over union.
    """
    iou_values = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if iou_values.size == 0:
        return iou_values

    iou_values = bbox_overlap(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float),
    )

    return iou_values


def iou_distance(atracks: List[Any], btracks: List[Any]) -> np.ndarray:
    """Compute cost based on Intersection over Union.

    Args:
        atracks (list[Any]): List of STracks.
        btracks (list[Any]): List of STracks.

    Returns:
        np.ndarray: Cost matrix of distance between intersection over
            union of bounding boxes.
    """
    if (atracks and isinstance(atracks[0], np.ndarray)) or (
        btracks and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    iou_values = ious(atlbrs, btlbrs)
    cost_matrix = 1 - iou_values

    return cost_matrix


# pylint: disable=unused-argument
def embedding_distance(
    tracks: List[Any], detections: List[Any], metric: str = "cosine"
) -> np.ndarray:
    """Calculate distance joint STracks and detections.

    Args:
        tracks (List[Any]): List of STracks.
        detections (List[Any]): List of STracks that are model predictions.
        metric (str): Type of distance metric to use. Defaults to "cosine".

    Returns:
        np.ndarray: Cost matrix of distance.
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    # Nomalized features
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features))

    return cost_matrix


# pylint: disable=too-many-arguments
def fuse_motion(
    kalman_filter: KalmanFilter,
    cost_matrix: np.ndarray,
    tracks: List[Any],
    detections: List[Any],
    only_position: bool = False,
    lambda_threshold: float = 0.98,
) -> np.ndarray:
    """Fuse various motions of detections and STracks.

    Args:
        kalman_filter (KalmanFilter): Kalman filter module.
        cost_matrix (np.ndarray): Cost matrix of embedded distances.
        tracks (List[Any]): Currently tracked_stracks and lost_stracks.
            Type is of List[STrack].
        detections (List[Any]): Predictions of model. Type is of List[STrack].
        only_position (bool): Gating dimension. Defaults to False.
        lambda_threshold (float): Threshold parameter. Defaults to 0.98.

    Returns:
        np.ndarray: Distances of fused motion.
    """
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kalman_filter.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric="maha"
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = (
            lambda_threshold * cost_matrix[row]
            + (1 - lambda_threshold) * gating_distance
        )
    return cost_matrix
