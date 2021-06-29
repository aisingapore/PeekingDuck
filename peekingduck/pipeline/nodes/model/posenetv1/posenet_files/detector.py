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
Core functions to use posenet to detect poses
"""


from typing import List
import numpy as np
import tensorflow as tf

from peekingduck.pipeline.nodes.model.posenetv1.posenet_files.decode_multi import \
    decode_multiple_poses


def get_keypoints_relative_coords(keypoint_coords: np.ndarray,
                                  output_scale: np.ndarray,
                                  image_size: List[int]) -> np.ndarray:
    """ Get keypoints coordinates relative to image size
    Args:
        keypoints_coords (np.array): Nx17x2 keypoints coordinates of N persons
        output_scale (np.array): output scale in Hx2 format
        image_size (List[int]): image size in HxW format
    Returns:
        keypoints_coords (np.array): Nx17x2 keypoints coordinates of N persons
                relative to image size
    """
    assert len(keypoint_coords.shape) == 3, "keypoint_coords should be 3D"
    assert keypoint_coords.shape[
        2], "keypoint_coords should be a 2D matrix of 2D offsets"
    rel_keypoint_coords = keypoint_coords * output_scale
    rel_keypoint_coords = rel_keypoint_coords / image_size
    return rel_keypoint_coords


def _sigmoid(array: np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-array))


def detect_keypoints(
        tf_model: tf.keras.Model,
        image: tf.Tensor,
        output_stride: int,
        dst_scores: np.ndarray,
        dst_keypoints: np.ndarray,
        model_type: str,
        score_threshold: float) -> int:
    # pylint: disable=too-many-arguments
    """ Evaluate image by model function to get detected keypoints
    Args:
        tf_model: tensorflow model
        image (tf.Tensor): image for inference
        output_stride (int): output stride to convert output indices to image coordinates
        dst_scores (np.array): Nx17 buffer to store keypoint scores where N is
            the max persons to be detected
        dst_keypoints (np.array): Nx17x2 buffer to store keypoints coordinate
            where N is the max persons to be detected
        model_type (str): specified model type (refer to modelconfig.yml)
        score_threshold (float): threshold for prediction
    Returns:
        pose_count (int): number of poses detected
    """
    model_output = tf_model(image)

    # For resnet's implementation, we need to apply a sigmoid function on
    # the heatmap, which is the first tensor in the output
    if model_type == 'resnet':
        model_output[0] = _sigmoid(model_output[0])

    pose_count = decode_multiple_poses(
        model_output,
        dst_scores,
        dst_keypoints,
        output_stride=output_stride,
        min_pose_score=score_threshold)

    return pose_count
