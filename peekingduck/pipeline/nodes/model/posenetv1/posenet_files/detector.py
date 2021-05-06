"""
Copyright 2018 Ross Wightman
Modifications copyright 2021 AI Singapore

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
from typing import List
import numpy as np

from peekingduck.pipeline.nodes.model.posenetv1.posenet_files.decode_multi import \
    decode_multiple_poses


def get_keypoints_relative_coords(keypoint_coords: List[float],
                                  output_scale: List[float],
                                  image_size: List[int]):
    """ Get relative coordinates that percentage of a keypoints to image size (W x H).
    It swaps array columns to change from (row, col) coordinate to (x, y) coordinate.


    Args:
        keypoints_coords (np.array): nx17x2 keypoints coordinates of n persons
        output_scale (np.array): output scale in hx2 format
        image_size (np.array): image size in hxw format

    Returns:
        keypoints_coords (np.array): nx17x2 keypoints coordinates of n persons
                relative to image size
    """
    assert len(keypoint_coords.shape) == 3, "keypoint_coords should be 3D"
    assert keypoint_coords.shape[
        2], "keypoint_coords should be a 2D matrix of 2D offsets"
    keypoint_coords *= output_scale
    keypoint_coords /= image_size
    return keypoint_coords


def _sigmoid(num: float):
    return 1/(1 + np.exp(-num))


def detect_keypoints(
        tf_model,
        image: List[List[float]],
        output_stride: int,
        dst_scores: List[float],
        dst_keypoints: List[List[float]],
        model_type: str,
        score_threshold: float):
    """ Evaluate image by model function to get keypoints info in yx format

    Args:
        tf_model: tensorflow model
        image (np.array): image for inference
        output_stride (int): output stride to convert output indices to image coordinates
        dst_scores (np.array): (nx17) buffer to store keypoint scores where n is
            the max persons to be detected
        dst_keypoints (np.array): (nx17x2) buffer to store keypoints coordinate
            where n is the max persons to be detected
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
