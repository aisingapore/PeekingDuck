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
Decoder class for decoding multiple poses
"""

import operator
from typing import List, Tuple, Union

import numpy as np
import scipy.ndimage as ndi
import tensorflow as tf

from peekingduck.nodes.model.posenetv1.posenet_files.constants import (
    LOCAL_MAXIMUM_RADIUS,
    MIN_ROOT_SCORE,
    NMS_RADIUS,
    POSE_CONNECTIONS,
)


class Decoder:  # pylint: disable=too-few-public-methods
    """Decodes PoseNet model output into multiple poses."""

    def __init__(self, pose_score_threshold: float, use_jit: bool) -> None:
        self.part_score_threshold = MIN_ROOT_SCORE
        self.pose_score_threshold = pose_score_threshold

        # pylint: disable=import-outside-toplevel
        if use_jit:
            from peekingduck.nodes.model.posenetv1.posenet_files import jit_funcs

            self._build_parts_with_peaks = jit_funcs.build_parts_with_peaks
            self._is_within_nms_radius = jit_funcs.is_within_nms_radius
            self._traverse_to_target_keypoint = jit_funcs.traverse_to_target_keypoint
        else:
            from peekingduck.nodes.model.posenetv1.posenet_files import nojit_funcs

            self._build_parts_with_peaks = nojit_funcs.build_parts_with_peaks
            self._is_within_nms_radius = nojit_funcs.is_within_nms_radius
            self._traverse_to_target_keypoint = nojit_funcs.traverse_to_target_keypoint

    def decode(
        self,
        model_output: List[tf.Tensor],
        pose_scores: np.ndarray,
        poses: np.ndarray,
        output_stride: int,
    ) -> int:
        """Decodes the offsets and displacements map and return the keypoints

        Args:
            model_output (List[tf.Tensor]): Scores, offsets, displacements_fwd,
                and displacements_bwd from model predictions.
            pose_scores (np.array): Nx17 buffer to store keypoint scores where N is
                the max persons to be detected.
            poses (np.array): Nx17x2 buffer to store keypoints coordinate where N
                is the max persons to be detected.
            output_stride (int): Output stride to convert output indices to image
                coordinates.

        Returns:
            pose_count (int): number of poses detected
        """
        scores = np.array(model_output[0][0])
        offsets = np.array(model_output[1][0])
        displacements_fwd = np.array(model_output[2][0])
        displacements_bwd = np.array(model_output[3][0])

        parts = self._build_parts(scores)

        (  # pylint: disable=unbalanced-tuple-unpacking
            offsets,
            displacements_fwd,
            displacements_bwd,
        ) = _change_dimensions(offsets, displacements_fwd, displacements_bwd)

        pose_count = self._look_for_poses(
            parts,
            scores,
            offsets,
            displacements_fwd,
            displacements_bwd,
            pose_scores,
            poses,
            output_stride,
        )

        return pose_count

    def _build_parts(self, scores: np.ndarray) -> List[Tuple[float, int, np.ndarray]]:
        """Builds the parts list. Each part in the list consists of the score,
        keypoint ID, and coordinates. A part is defined as the cell with the highest
        score in the local window of size (D, D, 1) where
        D = 2 * LOCAL_MAXIMUM_RADIUS + 1.

        Args:
            scores (np.ndarray): Score heatmap output by the model.

        Returns:
            (List[Tuple[float, int, np.ndarray]]): The parts list sorted by score in
            descending order.
        """
        diameter = 2 * LOCAL_MAXIMUM_RADIUS + 1

        local_peaks = ndi.maximum_filter(
            scores, size=(diameter, diameter, 1), mode="constant"
        )
        parts = self._build_parts_with_peaks(
            scores, self.part_score_threshold, local_peaks
        )
        parts = sorted(parts, key=operator.itemgetter(0), reverse=True)
        return parts

    def _calculate_instance_keypoints(  # pylint: disable=too-many-arguments
        self,
        edge: int,
        target_keypoint_id: int,
        source_keypoint_id: int,
        keypoint_scores: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        offsets: np.ndarray,
        output_stride: int,
        displacements: np.ndarray,
    ) -> None:
        """Obtain instance keypoints scores and coordinates"""
        if (
            keypoint_scores[source_keypoint_id] > 0.0
            and keypoint_scores[target_keypoint_id] == 0.0
        ):
            source_keypoint = keypoints[source_keypoint_id]

            score, coords = self._traverse_to_target_keypoint(
                edge,
                source_keypoint,
                target_keypoint_id,
                scores,
                offsets,
                output_stride,
                displacements,
            )

            keypoint_scores[target_keypoint_id] = score
            keypoints[target_keypoint_id] = coords

    def _decode_pose(  # pylint: disable=too-many-arguments
        self,
        root_score: float,
        root_id: int,
        root_image_coord: np.ndarray,
        scores: np.ndarray,
        offsets: np.ndarray,
        output_stride: int,
        displacements_fwd: np.ndarray,
        displacements_bwd: np.ndarray,
        keypoint_scores: np.ndarray,
        keypoints: np.ndarray,
    ) -> None:
        """Decode pose's keypoints scores and coordinates from keypoints score,
        coordinates and displacements

        Args:
            root_score (float): a keypoint with highest score is selected as root
            root_id (int): root keypoint's index
            root_image_coord (np.array): relative coordinate of root keypoint
            scores (np.array): HxWxNP heatmap scores of NP body parts
            offsets (np.array): HxWxNPx2 short range offset vector of NP body parts
            output_stride (int): output stride to convert output indices to image
                coordinates
            displacements_fwd (np.array): HxWxNEx2 forward displacements of NE body
                    connections
            displacements_bwd (np.array): HxWxNEx2 backward displacements of NE body
                    connections
            keypoints_scores (np.array): 17x1 buffer to store keypoint scores
            keypoint_coords (np.array): 17x2 buffer to store keypoint coordinates

        Returns:
            pose_count (int): number of poses detected
        """
        num_edges = len(POSE_CONNECTIONS)

        keypoint_scores[root_id] = root_score
        keypoints[root_id] = root_image_coord

        for edge in reversed(range(num_edges)):
            target_keypoint_id, source_keypoint_id = POSE_CONNECTIONS[edge]
            self._calculate_instance_keypoints(
                edge,
                target_keypoint_id,
                source_keypoint_id,
                keypoint_scores,
                keypoints,
                scores,
                offsets,
                output_stride,
                displacements_bwd,
            )

        for edge in range(num_edges):
            source_keypoint_id, target_keypoint_id = POSE_CONNECTIONS[edge]
            self._calculate_instance_keypoints(
                edge,
                target_keypoint_id,
                source_keypoint_id,
                keypoint_scores,
                keypoints,
                scores,
                offsets,
                output_stride,
                displacements_fwd,
            )

    def _look_for_poses(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        parts: List[Tuple[float, int, np.ndarray]],
        scores: np.ndarray,
        offsets: np.ndarray,
        displacements_fwd: np.ndarray,
        displacements_bwd: np.ndarray,
        pose_scores: np.ndarray,
        poses: np.ndarray,
        output_stride: int,
    ) -> int:
        """Change dimensions from (h, w, x) to (h, w, x//2, 2) to return a
        complete coordinate array
        """
        pose_count = 0
        max_pose_detections = pose_scores.shape[0]
        squared_nms_radius = NMS_RADIUS**2

        for root_score, root_id, root_coords in parts:
            root_image_coords = (
                root_coords * output_stride
                + offsets[root_coords[0], root_coords[1], root_id]
            )

            if self._is_within_nms_radius(
                poses[:pose_count, root_id], squared_nms_radius, root_image_coords
            ):
                continue

            keypoint_scores = pose_scores[pose_count]
            keypoints = poses[pose_count]
            self._decode_pose(
                root_score,
                root_id,
                root_image_coords,
                scores,
                offsets,
                output_stride,
                displacements_fwd,
                displacements_bwd,
                keypoint_scores,
                keypoints,
            )

            pose_score = _get_instance_score_fast(
                poses[:pose_count],
                NMS_RADIUS,
                keypoint_scores,
                keypoints,
            )
            if pose_score >= self.pose_score_threshold:
                pose_count += 1
            else:
                pose_scores[pose_count] = 0

            if pose_count >= max_pose_detections:
                break

        return pose_count


def _change_dimensions(*arrays: np.ndarray) -> Union[np.ndarray, List[np.ndarray]]:
    """Changes the dimensions of input arrays from (h, w, x) to (h, w, x//2, 2)
    to return a complete coordinate array.
    """
    results = []
    for array in arrays:
        height, width = array.shape[0], array.shape[1]
        results.append(array.reshape(height, width, 2, -1).swapaxes(2, 3))

    return results[0] if len(results) == 1 else results


def _get_instance_score_fast(
    existing_coords: np.ndarray,
    nms_radius: int,
    keypoint_scores: np.ndarray,
    keypoints: np.ndarray,
) -> float:
    """Obtain instance scores for keypoints"""
    if existing_coords.shape[0] > 0:
        score_sum = np.linalg.norm(existing_coords - keypoints, axis=2) > nms_radius
        not_overlapped_scores = np.sum(keypoint_scores[np.all(score_sum, axis=0)])
    else:
        not_overlapped_scores = np.sum(keypoint_scores)
    return not_overlapped_scores / len(keypoint_scores)
