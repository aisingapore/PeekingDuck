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
import scipy.ndimage as ndi
from .decode import decode_pose
LOCAL_MAXIMUM_RADIUS = 1

def decode_multiple_poses_xy(model_output,
                             dst_keypoint_scores,
                             dst_keypoints,
                             output_stride,
                             score_threshold=0.5,
                             nms_radius=20,
                             min_pose_score=0.5):
    """See _decode_multiple_poses(...) for details.

    It decodes the offsets and displacements map in xy order and return the
    keypoints in (x, y) format.

    args:
        - model_output: consist of 4 output maps from model predictions
            1. scores: 3-D tensor with shape [height, width, numParts], e.g. (33, 58, 17).
                The value of heatmapScores[y, x, k] is the score of placing the k-th
                object part at position (y, x).
            2. offsets: 4-D tensor with shape [height, width, numParts, 2],
                e.g. (33, 58, 17, 2).  The value of [offsets[y, x, k] is the short range
                offset vector of the `k`-th  object part at heatmap position (y, x).
            3. displacements_fwd: 4-D tensor with shape [height, width, num_edges, 2], where
                'num_edges = num_parts - 1', is the number of edges (parent-child pairs)
                in the tree. It contains the forward displacements between consecutive part
                from the root towards the leaves. e.g. (33, 58, 16, 2).
            4. displacements_bwd:  4-D tensor with shape [height, width, num_edges, 2], where
                'num_edges = num_parts - 1', is the number of edges (parent-child pairs)
                in the tree. It contains the backward displacements between consecutive part
                from the root towards the leaves. e.g. (33, 58, 16, 2).
        - dst_keypoint_scores: nx17 buffer to contain the detected keypoint scores.
                               n is max number of person to detect
        - dst_keypoints: nx17x2 buffer to contain the detected keypoint coordinates
                         n is max number of person to detect
        - output_stride: The output stride that was used when feed-forwarding
            through the PoseNet model.  Must be 32, 16, or 8.
        - score_threshold: Only return instance detections that have root part score
            greater or equal to this value. Defaults to 0.5.
        - nms_radius: Non-maximum suppression part distance. It needs to be strictly
            positive. Two parts suppress each other if they are less than `nmsRadius`
            pixels away. Defaults to 20.
        - min_pose_score: Minimum number of pose score of returned pose detection per image.

    return:
        number of poses detected. The keypoint scores and coordinates are in
        dst_keypoint_scores and dst_keypoints respectively
    """
    return _decode_multiple_poses(
        model_output,
        dst_keypoint_scores,
        dst_keypoints,
        output_stride,
        score_threshold,
        nms_radius,
        min_pose_score,
        decode_from_yx=False)


def decode_multiple_poses_yx(model_output,
                             dst_keypoint_scores,
                             dst_keypoints,
                             output_stride,
                             score_threshold=0.5,
                             nms_radius=20,
                             min_pose_score=0.5):
    """See _decode_multiple_poses(...) for details.

    This is same as decode_multiple_poses_xy() except when it decodes the
    offsets and displacements map in yx order and return the keypoints in (x, y)
    format.

    args:
        - model_output: consist of 4 output maps from model predictions
            1. scores: 3-D tensor with shape [height, width, numParts], e.g. (33, 58, 17).
                The value of heatmapScores[y, x, k] is the score of placing the k-th
                object part at position (y, x).
            2. offsets: 4-D tensor with shape [height, width, numParts, 2],
                e.g. (33, 58, 17, 2).  The value of [offsets[y, x, k] is the short range
                offset vector of the `k`-th  object part at heatmap position (y, x).
            3. displacements_fwd: 4-D tensor with shape [height, width, num_edges, 2], where
                'num_edges = num_parts - 1', is the number of edges (parent-child pairs)
                in the tree. It contains the forward displacements between consecutive part
                from the root towards the leaves. e.g. (33, 58, 16, 2).
            4. displacements_bwd:  4-D tensor with shape [height, width, num_edges, 2], where
                'num_edges = num_parts - 1', is the number of edges (parent-child pairs)
                in the tree. It contains the backward displacements between consecutive part
                from the root towards the leaves. e.g. (33, 58, 16, 2).
        - dst_keypoint_scores: nx17 buffer to contain the detected keypoint scores.
                               n is max number of person to detect
        - dst_keypoints: nx17x2 buffer to contain the detected keypoint coordinates
                         n is max number of person to detect
        - output_stride: The output stride that was used when feed-forwarding
            through the PoseNet model.  Must be 32, 16, or 8.
        - score_threshold: Only return instance detections that have root part score
            greater or equal to this value. Defaults to 0.5.
        - nms_radius: Non-maximum suppression part distance. It needs to be strictly
            positive. Two parts suppress each other if they are less than `nmsRadius`
            pixels away. Defaults to 20.
        - min_pose_score: Minimum number of pose score of returned pose detection per image.

    return:
        number of poses detected. The keypoint scores and coordinates are in
        dst_keypoint_scores and dst_keypoints respectively
    """
    return _decode_multiple_poses(
        model_output,
        dst_keypoint_scores,
        dst_keypoints,
        output_stride,
        score_threshold,
        nms_radius,
        min_pose_score,
        decode_from_yx=True)


def _decode_multiple_poses(model_output,
                           dst_keypoint_scores,
                           dst_keypoints,
                           output_stride,
                           score_threshold=0.5,
                           nms_radius=20,
                           min_pose_score=0.5,
                           decode_from_yx=False):
    """Decode multiple persons' poses from model outputs
    Taken from:
    https://medium.com/@prajwalsingh_48273/posenet-for-android-8b6dede9fa2f
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
        displacements_bwd,
        decode_from_yx)

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
        score_threshold, local_max_radius, scores):
    """Return an array of parts with score, id and coordinate in each part"""
    parts = []
    lmd = 2 * local_max_radius + 1

    max_vals = ndi.maximum_filter(scores, size=(lmd, lmd, 1), mode='constant')
    max_loc = np.logical_and(
        scores == max_vals, scores > score_threshold)
    max_loc_idx = max_loc.nonzero()
    for y, x, keypoint_id in zip(*max_loc_idx):
        parts.append((scores[y, x, keypoint_id],
                      keypoint_id,
                      np.array((x, y))))

    return parts


def _sort_scored_parts(parts):
    parts = sorted(parts, key=lambda x: x[0], reverse=True)
    return parts


def _change_dimensions(scores,
                       offsets,
                       displacements_fwd,
                       displacements_bwd,
                       decode_from_yx):
    # change dimensions from (h, w, x) to (h, w, x//2, 2) to allow return of
    # complete coord array
    height = scores.shape[0]
    width = scores.shape[1]
    offsets = offsets.reshape(height, width, 2, -1).swapaxes(2, 3)
    displacements_fwd = displacements_fwd.reshape(
        height, width, 2, -1).swapaxes(2, 3)
    displacements_bwd = displacements_bwd.reshape(
        height, width, 2, -1).swapaxes(2, 3)

    if decode_from_yx:
        SWAP_AXES = (1, 0)
        offsets = offsets[:, :, :, SWAP_AXES]
        displacements_fwd = displacements_fwd[:, :, :, SWAP_AXES]
        displacements_bwd = displacements_bwd[:, :, :, SWAP_AXES]

    return offsets, displacements_fwd, displacements_bwd


def _look_for_poses(scored_parts,
                    scores,
                    offsets,
                    displacements_fwd,
                    displacements_bwd,
                    dst_keypoint_scores,
                    dst_keypoints,
                    output_stride,
                    nms_radius,
                    min_pose_score):
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

        # NOTE this isn't in the original implementation, but it appears
        # that by initially ordering by part scores, and having a max #
        # of detections, we can end up populating the returned poses with
        # lower scored poses than if we discard 'bad' ones and continue
        # (higher pose scores can still come later).
        # Set min_pose_score to 0. to revert to original behaviour
        if min_pose_score == 0. or pose_score >= min_pose_score:
            pose_count += 1
        else:
            dst_keypoint_scores[pose_count] = 0

        if pose_count >= max_pose_detections:
            break

    return pose_count


def _calculate_keypoint_coords_on_image(heatmap_positions, output_stride, offsets, keypoint_id):
    """Formula from https://medium.com/@prajwalsingh_48273/posenet-for-android-8b6dede9fa2f

    keypoint_positions = heatmap_positions * output_stride + offset_vectors
    """
    offset_vectors = offsets[heatmap_positions[1],
                             heatmap_positions[0],
                             keypoint_id]
    return heatmap_positions * output_stride + offset_vectors


def _within_nms_radius_fast(pose_coords, squared_nms_radius, point):
    if not pose_coords.shape[0]:
        return False
    return np.any(
        np.sum((pose_coords - point)**2, axis=1) <= squared_nms_radius)


def _get_instance_score_fast(exist_pose_coords, squared_nms_radius,
                             keypoint_scores, keypoint_coords):

    if exist_pose_coords.shape[0]:
        s = np.sum((exist_pose_coords - keypoint_coords) ** 2, axis=2)\
            > squared_nms_radius
        not_overlapped_scores = np.sum(keypoint_scores[np.all(s, axis=0)])
    else:
        not_overlapped_scores = np.sum(keypoint_scores)
    return not_overlapped_scores / len(keypoint_scores)
