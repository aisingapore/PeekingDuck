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
Supporting functions to decode multiple poses
"""

import operator
from typing import List, Tuple

import numpy as np
import scipy.ndimage as ndi
import tensorflow as tf

from peekingduck.nodes.model.posenetv1.posenet_files.constants import (
    LOCAL_MAXIMUM_RADIUS,
    SWAP_AXES,
)
from peekingduck.nodes.model.posenetv1.posenet_files.decode import decode_pose


def decode_multiple_poses(
    model_output: List[tf.Tensor],
    dst_keypoint_scores: np.ndarray,
    dst_keypoints: np.ndarray,
    output_stride: int,
    score_threshold: float = 0.5,
    nms_radius: int = 20,
    min_pose_score: float = 0.5,
) -> int:
    # pylint: disable=too-many-arguments
    """Decodes the offsets and displacements map and return the keypoints


    Args:
        model_output (List[tf.Tensor]): Scores, offsets, displacements_fwd, and
            displacements_bwd from model predictions.
        dst_scores (np.array): Nx17 buffer to store keypoint scores where N is
                        the max persons to be detected
        dst_keypoints (np.array): Nx17x2 buffer to store keypoints coordinate
                        where N is the max persons to be detected
        output_stride (int): output stride to convert output indices to image coordinates
        score_threshold (float): return instance detections if root part score
                        >= threshold
        nms_radius (int): non-maximum suppression part distance in pixels
        min_pose_score (float): minimum pose score to return detected pose

    Returns:
        pose_count (int): number of poses detected
    """
    scores = np.array(model_output[0][0])
    offsets = np.array(model_output[1][0])
    displacements_fwd = np.array(model_output[2][0])
    displacements_bwd = np.array(model_output[3][0])

    parts = _build_parts_with_score(scores, score_threshold)
    # Sort parts by score in descending order
    parts = sorted(parts, key=operator.itemgetter(0), reverse=True)

    offsets, displacements_fwd, displacements_bwd = _change_dimensions(
        offsets, displacements_fwd, displacements_bwd
    )

    pose_count = _look_for_poses(
        parts,
        scores,
        offsets,
        displacements_fwd,
        displacements_bwd,
        dst_keypoint_scores,
        dst_keypoints,
        output_stride,
        nms_radius,
        min_pose_score,
    )

    return pose_count


def _build_parts_with_score(
    scores: np.ndarray, score_threshold: float
) -> List[Tuple[float, int, np.ndarray]]:
    """Builds the parts list. Each part in the list consists of the score,
    keypoint ID, and coordinates. A part is defined as the cell with the highest
    score in the local window of size (2 x LOCAL_MAXIMUM_RADIUS, 2 x LOCAL_MAXIMUM_RADIUS, 1).

    Args:
        scores (np.ndarray): Score heatmap output by the model.
        score_threshold (float): Local peaks below this threshold will be
            discarded.
    """
    parts = []
    diameter = 2 * LOCAL_MAXIMUM_RADIUS + 1

    local_peaks = ndi.maximum_filter(
        scores, size=(diameter, diameter, 1), mode="constant"
    )
    local_peak_locations = (
        (scores == local_peaks) & (scores > score_threshold)
    ).nonzero()
    for y_coord, x_coord, keypoint_id in zip(*local_peak_locations):
        parts.append(
            (
                scores[y_coord, x_coord, keypoint_id],
                keypoint_id,
                np.array((x_coord, y_coord)),
            )
        )

    return parts


def _calculate_keypoint_coords_on_image(
    heatmap_positions: np.ndarray,
    output_stride: int,
    offsets: np.ndarray,
    keypoint_id: int,
) -> np.ndarray:
    """Calculate keypoint image coordinates from heatmap positions,
    output_stride and offset_vectors
    """
    offset_vectors = offsets[heatmap_positions[1], heatmap_positions[0], keypoint_id]
    return heatmap_positions * output_stride + offset_vectors


def _change_dimensions(
    offsets: np.ndarray,
    displacements_fwd: np.ndarray,
    displacements_bwd: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Change dimensions from (h, w, x) to (h, w, x//2, 2) to return a
    complete coordinate array
    """
    height, width = offsets.shape[0], offsets.shape[1]
    offsets = offsets.reshape(height, width, 2, -1).swapaxes(2, 3)
    displacements_fwd = displacements_fwd.reshape(height, width, 2, -1).swapaxes(2, 3)
    displacements_bwd = displacements_bwd.reshape(height, width, 2, -1).swapaxes(2, 3)

    offsets = offsets[:, :, :, SWAP_AXES]
    displacements_fwd = displacements_fwd[:, :, :, SWAP_AXES]
    displacements_bwd = displacements_bwd[:, :, :, SWAP_AXES]

    return offsets, displacements_fwd, displacements_bwd


def _get_instance_score_fast(
    existing_coords: np.ndarray,
    squared_nms_radius: int,
    keypoint_scores: np.ndarray,
    keypoint_coords: np.ndarray,
) -> float:
    """Obtain instance scores for keypoints"""
    if existing_coords.shape[0] > 0:
        score_sum = (
            np.sum((existing_coords - keypoint_coords) ** 2, axis=2)
            > squared_nms_radius
        )
        not_overlapped_scores = np.sum(keypoint_scores[np.all(score_sum, axis=0)])
    else:
        not_overlapped_scores = np.sum(keypoint_scores)
    return not_overlapped_scores / len(keypoint_scores)


def _is_within_nms_radius(
    existing_coords: np.ndarray, radius: int, point: np.ndarray
) -> bool:
    """Checks if the specified `point` is within the radius of any existing
    coordinates.

    Args:
        existing_coords (np.ndarray): Array of existing coordinates.
        radius (int): The radius/distance to check for.
        point (np.ndarray): The specified point to check.
    """
    return existing_coords.shape[0] > 0 and np.any(
        np.sum((existing_coords - point) ** 2, axis=1) <= radius
    )


def _look_for_poses(
    parts: List[Tuple[float, int, np.ndarray]],
    scores: np.ndarray,
    offsets: np.ndarray,
    displacements_fwd: np.ndarray,
    displacements_bwd: np.ndarray,
    dst_keypoint_scores: np.ndarray,
    dst_keypoints: np.ndarray,
    output_stride: int,
    nms_radius: int,
    min_pose_score: float,
) -> int:
    # pylint: disable=too-many-arguments, too-many-locals
    """Change dimensions from (h, w, x) to (h, w, x//2, 2) to return a
    complete coordinate array
    """
    pose_count = 0
    dst_keypoint_scores[:] = 0
    max_pose_detections = dst_keypoint_scores.shape[0]
    squared_nms_radius = nms_radius ** 2

    for root_score, root_id, root_coord in parts:
        root_image_coords = _calculate_keypoint_coords_on_image(
            root_coord, output_stride, offsets, root_id
        )

        if _is_within_nms_radius(
            dst_keypoints[:pose_count, root_id, :],
            squared_nms_radius,
            root_image_coords,
        ):
            continue

        keypoint_scores = dst_keypoint_scores[pose_count]
        keypoint_coords = dst_keypoints[pose_count]
        decode_pose(
            root_score,
            root_id,
            root_image_coords,
            scores,
            offsets,
            output_stride,
            displacements_fwd,
            displacements_bwd,
            keypoint_scores,
            keypoint_coords,
        )

        pose_score = _get_instance_score_fast(
            dst_keypoints[:pose_count, :, :],
            squared_nms_radius,
            keypoint_scores,
            keypoint_coords,
        )
        if pose_score >= min_pose_score:
            pose_count += 1
        else:
            dst_keypoint_scores[pose_count] = 0

        if pose_count >= max_pose_detections:
            break

    return pose_count
