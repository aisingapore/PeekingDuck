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

"""
hod is short for heatmap, offsets and displacements.

This module detects keypoints from heatmap, offsets and displacements maps
"""
import tensorflow as tf
import numpy as np

# from models.pose.bottomup.utils.inference.hod.decode_multi import \
#     decode_multiple_poses_xy, decode_multiple_poses_yx
from .decode_multi import \
    decode_multiple_poses_xy, decode_multiple_poses_yx

SCALE_FACTOR = 1.0
KEYPOINTS_NUM = 17


def get_keypoints_relative_coords(keypoint_coords, output_scale, image_size):
    """Get relative coordinates that percentage of a keypoints to image size (W x H).
    It swaps array columns to change from (row, col) coordinate to (x, y) coordinate.

    args:
        - output_scale: (np.array) a list of [a, b] e.g. [1.4 1.4]
        - image_size:   (np.array) image size e.g. [1280, 720]

    return:
        - keypoint_coords: (n x 17 x 2) keypoints relative coordinate
                            n is the number of person assigned
                            before model created. e.g. [0.5, 0.7]
    """
    assert len(keypoint_coords.shape) == 3, "keypoint_coords should be 3D"
    assert keypoint_coords.shape[
        2], "keypoint_coords should be a 2D matrix of 2D offsets"
    keypoint_coords *= output_scale
    keypoint_coords /= image_size
    return keypoint_coords


def detect_keypoints_xy(
        tf_model, image, output_stride, dst_scores, dst_keypoints,
        score_threshold):
    """Evaluate image by model function to get keypoints info in xy format
    args:
        - model_func: the callable tensorflow model
        - image: (np.array) input image
        - dst_scores: (n x 17) buffer to store keypoint scores
                      n is max number of person to be detected
        - dst_keypoints: (n x 17 x 2) buffer to store keypoints coordinate
                         n is max number of person to be detected
        - score_threshold - the threshold for predictions

    return:
        number of poses detected
    """
    model_output = tf_model(
        tf.convert_to_tensor([image]), training=False)

    pose_count = decode_multiple_poses_xy(
        model_output,
        dst_scores,
        dst_keypoints,
        output_stride=output_stride,
        min_pose_score=score_threshold)

    return pose_count


def _sigmoid(x):
    return 1/(1 + np.exp(-x))


def detect_keypoints_yx(
        tf_model, image, output_stride, dst_scores, dst_keypoints, model_type,
        score_threshold):
    """Evaluate image by model function to get keypoints info in yx format

    This mainly exists for the tf 1.0 converted model, namely posenet.
    It should be removed if we can replace this model.

    Most of the rendering apis requires keypoint in xy format. Using
    this mean there is a need for an extra step to swap the y and x values.

    args:
        - model_func: the callable tensorflow model
        - image: (np.array) input image
        - dst_scores: (n x 17) buffer to store keypoint scores
                      n is max number of person to be detected
        - dst_keypoints: (n x 17 x 2) buffer to store  keypoints coordinate
                         n is max number of person to be detected
        - score_threshold - the threshold for predictions

    return:
        number of poses detected
    """
    model_output = tf_model(image)

    # For resnet's implementation, we need to use a sigmoid function on
    # the heatmap, which is the first tensor in the output

    # TODO: implement sigmoid as the final layer of the model/graph instead
    # if possible. the implementation now is from tensorflow's official
    # git repo. It can be found here
    # https://github.com/tensorflow/tfjs-models/tree/master/posenet
    if model_type == 'resnet':
        model_output[0] = _sigmoid(model_output[0])

    pose_count = decode_multiple_poses_yx(
        model_output,
        dst_scores,
        dst_keypoints,
        output_stride=output_stride,
        min_pose_score=score_threshold)

    return pose_count
