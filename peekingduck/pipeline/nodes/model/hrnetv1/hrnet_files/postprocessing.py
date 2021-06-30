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

"""
Postprocessing functions for HRNet
"""


from typing import List, Tuple
import numpy as np

SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4],
            [3, 5], [4, 6], [5, 7]]


def scale_transform(keypoints: np.ndarray,
                    in_scale: List[int],
                    out_scale: List[int]) -> np.ndarray:
    """Transform points from input scale to out scale.

    Args:
        keypoints (np.array): array of detected keypoints
        in_scale (list): the input scale of xy
        out_scale (list): the output scale of xy

    Returns:
        the scaled points from input scale to output scale
    """
    in_to_out_scale = np.array(out_scale) / np.array(in_scale)
    keypoints = keypoints * in_to_out_scale
    return keypoints


def affine_transform_xy(keypoints: np.ndarray, affine_matrices: np.ndarray) -> np.ndarray:
    """Apply respective affine transform on array of points.

    Args:
        keypoints (np.array): array sequence of points (x, y)
        affine_matrices (np.array) - array of 2x3 affine transform matrix

    Returns:
        array of transformed points
    """
    transformed_matrices = []
    keypoints = np.dstack((keypoints, np.ones((keypoints.shape[0], keypoints.shape[1], 1))))
    for affine_matrix, keypoint in zip(affine_matrices, keypoints):
        transformed_keypoint = np.dot(affine_matrix, keypoint.T)
        transformed_matrices.append(transformed_keypoint.T)

    return np.array(transformed_matrices)


def reshape_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
    """Helper function to reshape heatmaps to required shape

    Args:
        heatmaps (np.ndarray): Outputs heatmaps from hrnet network

    Returns:
        np.ndarray: reshaped heatmaps
    """
    batch, _, _, num_joints = heatmaps.shape

    heatmaps_reshaped = np.transpose(heatmaps, axes=(0, 3, 1, 2))
    heatmaps_reshaped = heatmaps_reshaped.reshape((batch, num_joints, -1))

    return heatmaps_reshaped


def get_valid_keypoints(keypoints: np.ndarray,
                        keypoint_scores: np.ndarray,
                        batch: int,
                        min_score: int) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function to get visible keypoints

    Args:
        keypoints (np.ndarray): array of detected keypoints
        keypoint_scores (np.ndarray): array of keypoint scores
        batch (int): number of detected bboxes
        min_score (int): min score threshold

    Returns:
        Tuple[np.ndarray, np.ndarray]: array of keypoints above threshold and keypoint mask
    """
    score_masks = keypoint_scores > min_score
    kp_masks = np.repeat(keypoint_scores > 0., 2).reshape(batch, 17, -1).astype(np.float32)
    keypoints *= kp_masks
    return keypoints, score_masks


def get_keypoint_conns(rel_keypoints: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Helper function to get keypoint connections

    Args:
        rel_keypoints (np.ndarray): Array of normalized keypoints
        masks (np.ndarray): Array of keypoint masks

    Returns:
        np.ndarray: Array of keypoint connections
    """
    compiled_connections = []
    for keypoint, mask in zip(rel_keypoints, masks):
        single_conn = _get_keypoint_of_single_pose(keypoint, mask)
        compiled_connections.append(single_conn)
    return np.array(compiled_connections, dtype=object)


def _get_keypoint_of_single_pose(keypoint: np.ndarray, mask: np.ndarray) -> np.ndarray:
    connections = []
    for start_joint, end_joint in SKELETON:
        if mask[start_joint - 1] and mask[end_joint - 1]:
            connections.append((keypoint[start_joint - 1], keypoint[end_joint - 1]))
    return np.array(connections)
