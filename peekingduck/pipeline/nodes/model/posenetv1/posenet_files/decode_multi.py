# Copyright 2018 Ross Wightman
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

"""
Supporting functions to decode multiple poses
"""


from typing import List, Tuple

import numpy as np
import tensorflow as tf
import scipy.ndimage as ndi
from peekingduck.pipeline.nodes.model.posenetv1.posenet_files.decode import decode_pose
from peekingduck.pipeline.nodes.model.posenetv1.posenet_files.constants import \
    LOCAL_MAXIMUM_RADIUS, SWAP_AXES


def decode_multiple_poses(model_output: Tuple[np.ndarray, tf.Tensor, tf.Tensor, tf.Tensor],
                          dst_keypoint_scores: np.ndarray,
                          dst_keypoints: np.ndarray,
                          output_stride: int,
                          score_threshold: float = 0.5,
                          nms_radius: int = 20,
                          min_pose_score: float = 0.5) -> int:
    # pylint: disable=too-many-arguments
    """ Decodes the offsets and displacements map and return the keypoints


    Args:
        model_output (Tuple[np.ndarray, tf.Tensor, tf.Tensor, tf.Tensor]): scores,
                offsets, displacements_fwd, displacements_bwd from model predictions
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
    scores, offsets, displacements_fwd, displacements_bwd = model_output
    scores = np.array(scores[0])
    offsets = np.array(offsets[0])
    displacements_fwd = np.array(displacements_fwd[0])
    displacements_bwd = np.array(displacements_bwd[0])

    scored_parts = _build_part_with_score_fast(score_threshold,
                                               LOCAL_MAXIMUM_RADIUS,
                                               scores)
    scored_parts = _sort_scored_parts(scored_parts)

    offsets, displacements_fwd, displacements_bwd = _change_dimensions(
        scores,
        offsets,
        displacements_fwd,
        displacements_bwd)

    pose_count = _look_for_poses(
        scored_parts,
        scores,
        offsets,
        displacements_fwd,
        displacements_bwd,
        dst_keypoint_scores,
        dst_keypoints,
        output_stride,
        nms_radius,
        min_pose_score)

    return pose_count


def _build_part_with_score_fast(
        score_threshold: float,
        local_max_radius: int,
        scores: np.ndarray) -> List[Tuple[float, int, np.ndarray]]:
    """ Returns an array of parts with score, id and coordinate in each part """
    parts = []
    lmd = 2 * local_max_radius + 1

    max_vals = ndi.maximum_filter(scores, size=(lmd, lmd, 1), mode='constant')
    max_loc = np.logical_and(
        scores == max_vals, scores > score_threshold)
    max_loc_idx = max_loc.nonzero()
    for y_coord, x_coord, keypoint_id in zip(*max_loc_idx):
        parts.append((scores[y_coord, x_coord, keypoint_id],
                      keypoint_id,
                      np.array((x_coord, y_coord))))

    return parts


def _sort_scored_parts(
        parts: List[Tuple[float, int, np.ndarray]]) -> List[Tuple[float, int, np.ndarray]]:
    """ Sort parts by confidence scores """
    parts = sorted(parts, key=lambda x: x[0], reverse=True)
    return parts


def _change_dimensions(scores: np.ndarray,
                       offsets: np.ndarray,
                       displacements_fwd: np.ndarray,
                       displacements_bwd: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Change dimensions from (h, w, x) to (h, w, x//2, 2) to return a
    complete coordinate array
    """
    height = scores.shape[0]
    width = scores.shape[1]
    offsets = offsets.reshape(height, width, 2, -1).swapaxes(2, 3)
    displacements_fwd = displacements_fwd.reshape(
        height, width, 2, -1).swapaxes(2, 3)
    displacements_bwd = displacements_bwd.reshape(
        height, width, 2, -1).swapaxes(2, 3)

    offsets = offsets[:, :, :, SWAP_AXES]
    displacements_fwd = displacements_fwd[:, :, :, SWAP_AXES]
    displacements_bwd = displacements_bwd[:, :, :, SWAP_AXES]

    return offsets, displacements_fwd, displacements_bwd


def _look_for_poses(scored_parts: List[Tuple[float, int, np.ndarray]],
                    scores: np.ndarray,
                    offsets: np.ndarray,
                    displacements_fwd: np.ndarray,
                    displacements_bwd: np.ndarray,
                    dst_keypoint_scores: np.ndarray,
                    dst_keypoints: np.ndarray,
                    output_stride: int,
                    nms_radius: int,
                    min_pose_score: float) -> int:
    # pylint: disable=too-many-arguments, too-many-locals
    """ Change dimensions from (h, w, x) to (h, w, x//2, 2) to return a
    complete coordinate array
    """
    pose_count = 0
    dst_keypoint_scores[:] = 0
    max_pose_detections = dst_keypoint_scores.shape[0]
    squared_nms_radius = nms_radius**2

    for root_score, root_id, root_coord in scored_parts:
        root_image_coords = _calculate_keypoint_coords_on_image(root_coord,
                                                                output_stride,
                                                                offsets,
                                                                root_id)

        if _within_nms_radius_fast(dst_keypoints[:pose_count, root_id, :],
                                   squared_nms_radius,
                                   root_image_coords):
            continue

        keypoint_scores = dst_keypoint_scores[pose_count]
        keypoint_coords = dst_keypoints[pose_count]
        decode_pose(root_score,
                    root_id,
                    root_image_coords,
                    scores,
                    offsets,
                    output_stride,
                    displacements_fwd,
                    displacements_bwd,
                    keypoint_scores,
                    keypoint_coords)

        pose_score = _get_instance_score_fast(dst_keypoints[:pose_count, :, :],
                                              squared_nms_radius,
                                              keypoint_scores,
                                              keypoint_coords)
        if min_pose_score == 0. or pose_score >= min_pose_score:
            pose_count += 1
        else:
            dst_keypoint_scores[pose_count] = 0

        if pose_count >= max_pose_detections:
            break

    return pose_count


def _calculate_keypoint_coords_on_image(heatmap_positions: np.ndarray,
                                        output_stride: int,
                                        offsets: np.ndarray,
                                        keypoint_id: int) -> np.ndarray:
    """ Calculate keypoint image coordinates from heatmap positions,
        output_stride and offset_vectors
    """
    offset_vectors = offsets[heatmap_positions[1],
                             heatmap_positions[0],
                             keypoint_id]
    return heatmap_positions * output_stride + offset_vectors


def _within_nms_radius_fast(pose_coords: np.ndarray,
                            squared_nms_radius: int,
                            point: np.ndarray) -> bool:
    """ Check if keypoint is within squared nms radius
    """
    if not pose_coords.shape[0]:
        return False
    return np.any(
        np.sum((pose_coords - point)**2, axis=1) <= squared_nms_radius)


def _get_instance_score_fast(exist_pose_coords: np.ndarray,
                             squared_nms_radius: int,
                             keypoint_scores: np.ndarray,
                             keypoint_coords: np.ndarray) -> float:
    """ Obtain instance scores for keypoints
    """
    if exist_pose_coords.shape[0]:
        score_sum = np.sum((exist_pose_coords - keypoint_coords) ** 2, axis=2)\
            > squared_nms_radius
        not_overlapped_scores = np.sum(keypoint_scores[np.all(score_sum, axis=0)])
    else:
        not_overlapped_scores = np.sum(keypoint_scores)
    return not_overlapped_scores / len(keypoint_scores)
