"""
Copyright 2021 AI Singapore

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np

from .constants import POSE_CONNECTIONS


def decode_pose(root_score,
                root_id,
                root_image_coord,
                scores,
                offsets,
                output_stride,
                displacements_fwd,
                displacements_bwd,
                keypoint_scores,
                keypoint_coords):
    """Decode pose's keypoints scores and coordinates from keypoints score,
    coordinates and displancements. An explantion of the algorithm is here:
    https://medium.com/@prajwalsingh_48273/posenet-for-android-8b6dede9fa2f

    args:
        - root_score: a keypoint with highest score is selected as root
        - root_id: root keypoint's index
        - root_image_coord: relative coordinate the root in image
        - scores: reference to decode_multiple_poses()
        - offsets: reference to decode_multiple_poses()
        - output_stride: reference to decode_multiple_poses()
        - displacements_fwd: reference to decode_multiple_poses()
        - displacements_bwd: reference to decode_multiple_poses()
        - keypoint_scores - 17x1 buffer to store keypoint scores
        - keypoint_coords - 17 x 2 buffer to store keypoint coordinates
    """
    num_edges = len(POSE_CONNECTIONS)

    keypoint_scores[root_id] = root_score
    keypoint_coords[root_id] = root_image_coord

    for edge in reversed(range(num_edges)):
        target_keypoint_id, source_keypoint_id = POSE_CONNECTIONS[edge]
        _calculate_instance_keypoints(edge,
                                      target_keypoint_id,
                                      source_keypoint_id,
                                      keypoint_scores,
                                      keypoint_coords,
                                      scores,
                                      offsets,
                                      output_stride,
                                      displacements_bwd)

    for edge in range(num_edges):
        source_keypoint_id, target_keypoint_id = POSE_CONNECTIONS[edge]
        _calculate_instance_keypoints(edge,
                                      target_keypoint_id,
                                      source_keypoint_id,
                                      keypoint_scores,
                                      keypoint_coords,
                                      scores,
                                      offsets,
                                      output_stride,
                                      displacements_fwd)


def _calculate_instance_keypoints(edge,
                                  target_keypoint_id,
                                  source_keypoint_id,
                                  instance_keypoint_scores,
                                  instance_keypoint_coords,
                                  scores,
                                  offsets,
                                  output_stride,
                                  displacements):
    if (instance_keypoint_scores[source_keypoint_id] > 0.0 and
            instance_keypoint_scores[target_keypoint_id] == 0.0):
        source_keypoint = instance_keypoint_coords[source_keypoint_id]

        score, coords = _traverse_to_target_keypoint(edge,
                                                     source_keypoint,
                                                     target_keypoint_id,
                                                     scores,
                                                     offsets,
                                                     output_stride,
                                                     displacements)

        instance_keypoint_scores[target_keypoint_id] = score
        instance_keypoint_coords[target_keypoint_id] = coords


def _clip_to_indices(keypoints, output_stride, width, height):
    """Clip keypoint coordinate to indices within dimension (width, height)"""
    keypoints = keypoints / output_stride
    keypoint_indices = np.zeros((2,), dtype=np.int32)

    keypoint_indices[0] = max(min(round(keypoints[0]), width - 1), 0)
    keypoint_indices[1] = max(min(round(keypoints[1]), height - 1), 0)

    return keypoint_indices


def _traverse_to_target_keypoint(edge_id,
                                 source_keypoint,
                                 target_keypoint_id,
                                 scores,
                                 offsets,
                                 output_stride,
                                 displacements):
    height = scores.shape[0] - 1
    width = scores.shape[1] - 1

    source_keypoint_indices = _clip_to_indices(
        source_keypoint, output_stride, width, height)

    displaced_point = source_keypoint + displacements[
        source_keypoint_indices[1], source_keypoint_indices[0], edge_id]

    displaced_point_indices = _clip_to_indices(
        displaced_point, output_stride, width, height)

    score = scores[displaced_point_indices[1],
                   displaced_point_indices[0], target_keypoint_id]

    image_coord = displaced_point_indices * output_stride + offsets[
        displaced_point_indices[1], displaced_point_indices[0], target_keypoint_id]

    return score, image_coord
