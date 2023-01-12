# Copyright 2018 Ross Wightman
#
# Modifications copyright 2022 AI Singapore
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

"""
PoseNet auxiliary functions (non-compiled)
"""

from typing import List, Tuple, cast

import numpy as np


def build_parts_with_peaks(
    scores: np.ndarray, score_threshold: float, local_peaks: np.ndarray
) -> List[Tuple[float, int, np.ndarray]]:
    """Builds the parts list. Each part in the list consists of the score,
    keypoint ID, and coordinates. A part is defined as the cell with the highest
    score in the local window of size (D, D, 1) where
    D = 2 * LOCAL_MAXIMUM_RADIUS + 1.

    Args:
        scores (np.ndarray): Score heatmap output by the model.
        score_threshold (float): Local peaks below this threshold will be
            discarded.
        local_peaks (np.ndarray): Local peaks in the score heatmap.

    Returns:
        (List[Tuple[float, int, np.ndarray]]): The parts list sorted by score in
        descending order.
    """
    parts = []
    local_peak_locations = (
        (scores == local_peaks) & (scores > score_threshold)
    ).nonzero()
    for y_coord, x_coord, keypoint_id in zip(*local_peak_locations):
        parts.append(
            (
                scores[y_coord, x_coord, keypoint_id],
                keypoint_id,
                np.array((y_coord, x_coord)),
            )
        )

    return parts


def is_within_nms_radius(
    existing_coords: np.ndarray, radius: int, point: np.ndarray
) -> bool:
    """Checks if the specified `point` is within the radius of any existing
    coordinates.

    Args:
        existing_coords (np.ndarray): Array of existing coordinates.
        radius (int): The radius/distance to check for.
        point (np.ndarray): The specified point to check.
    """
    return cast(
        bool,
        existing_coords.shape[0] > 0
        and np.any(np.sum((existing_coords - point) ** 2, axis=1) <= radius),
    )


def traverse_to_target_keypoint(  # pylint: disable=too-many-arguments
    edge_id: int,
    source_keypoint: np.ndarray,
    target_keypoint_id: int,
    scores: np.ndarray,
    offsets: np.ndarray,
    output_stride: int,
    displacements: np.ndarray,
) -> Tuple[float, float]:
    """Traverse to target keypoint to obtain keypoint score and coordinates"""
    height = scores.shape[0] - 1
    width = scores.shape[1] - 1

    source_keypoint_indices = _clip_to_indices(
        source_keypoint, output_stride, width, height
    )

    displaced_point = (
        source_keypoint
        + displacements[source_keypoint_indices[0], source_keypoint_indices[1], edge_id]
    )

    displaced_point_indices = _clip_to_indices(
        displaced_point, output_stride, width, height
    )

    score = scores[
        displaced_point_indices[0], displaced_point_indices[1], target_keypoint_id
    ]

    image_coord = (
        displaced_point_indices * output_stride
        + offsets[
            displaced_point_indices[0], displaced_point_indices[1], target_keypoint_id
        ]
    )

    return score, image_coord


def _clip_to_indices(
    keypoints: np.ndarray, output_stride: int, width: int, height: int
) -> np.ndarray:
    """Clip keypoint coordinate to indices within dimension (width, height)"""
    keypoints = np.rint(keypoints / output_stride)
    keypoint_indices = np.zeros((2,), dtype=np.int32)

    keypoint_indices[0] = max(min(keypoints[0], width - 1), 0)
    keypoint_indices[1] = max(min(keypoints[1], height - 1), 0)

    return keypoint_indices
