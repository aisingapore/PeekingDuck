# Copyright 2022 AI Singapore
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
Predictor class to handle detection of poses for posenet
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf

from peekingduck.nodes.model.posenetv1.posenet_files.constants import (
    IMAGE_NET_MEAN,
    MIN_PART_SCORE,
)
from peekingduck.nodes.model.posenetv1.posenet_files.decode_multi import (
    decode_multiple_poses,
)
from peekingduck.utils.graph_functions import load_graph
from peekingduck.utils.pose.keypoint_handler import COCOBody

OUTPUT_STRIDE = 16


class Predictor:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """Predictor class to handle detection of poses for posenet"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model_dir: Path,
        model_type: Union[int, str],
        model_file: Dict[Union[int, str], str],
        model_nodes: Dict[str, Dict[str, List[str]]],
        resolution: Dict[str, int],
        max_pose_detection: int,
        score_threshold: float,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        self.model_type = model_type
        self.model_path = model_dir / model_file[self.model_type]
        self.model_nodes = model_nodes[
            "resnet" if self.model_type == "resnet" else "mobilenet"
        ]

        self.resolution = int(resolution["height"]), int(resolution["width"])
        self.max_pose_detection = max_pose_detection
        self.score_threshold = score_threshold

        self.keypoint_handler = COCOBody(score_threshold=MIN_PART_SCORE)
        self.posenet = self._create_posenet_model()

    def predict(  # pylint: disable=too-many-locals
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """PoseNet prediction function

        Args:
            frame (np.array): image in numpy array

        Returns:
            bboxes (np.ndarray): array of bboxes
            keypoints (np.ndarray): array of keypoint coordinates
            keypoints_scores (np.ndarray): array of keypoint scores
            keypoints_conns (np.ndarray): array of keypoint connections
        """
        raw_keypoints, raw_keypoint_scores = self._predict_all_poses(frame)
        self.keypoint_handler.update(raw_keypoints, raw_keypoint_scores)

        if len(self.keypoint_handler.bboxes) == 0:
            return np.empty((0, 4)), np.empty(0), np.empty(0), np.empty(0)
        return (
            self.keypoint_handler.bboxes,
            self.keypoint_handler.keypoints,
            self.keypoint_handler.scores,
            self.keypoint_handler.connections,
        )

    def _create_posenet_model(self) -> Callable:
        self.logger.info(
            "PoseNet model loaded with following configs:\n\t"
            f"Model type: {self.model_type},\n\t"
            f"Input resolution: {self.resolution},\n\t"
            f"Max pose detection: {self.max_pose_detection},\n\t"
            f"Score threshold: {self.score_threshold}"
        )
        return self._load_posenet_weights()

    def _get_valid_resolution(self, output_stride: int) -> Tuple[int, int]:
        """Calculates the valid height and width divisible by `output_stride`."""
        target_height = int(self.resolution[0] / output_stride) * output_stride + 1
        target_width = int(self.resolution[1] / output_stride) * output_stride + 1
        return target_height, target_width

    def _load_posenet_weights(self) -> Callable:
        if not self.model_path.is_file():
            raise ValueError(
                f"Graph file does not exist. Please check that {self.model_path} exists"
            )
        return load_graph(
            str(self.model_path),
            inputs=self.model_nodes["inputs"],
            outputs=self.model_nodes["outputs"],
        )

    def _predict_all_poses(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict relative coordinates, confident scores and validation masks
        for all detected poses

        Args:
            frame (np.ndarray): image for inference

        Returns:
            full_keypoint_coords (np.ndarray): keypoints coordinates of
                detected poses
            full_keypoint_scores (np.ndarray): keypoints confidence scores of
                detected poses
        """
        # image_size = [W, H]
        image, output_scale, image_size = self._preprocess(frame)

        keypoint_scores = np.zeros((self.max_pose_detection, COCOBody.NUM_KEYPOINTS))
        keypoints = np.zeros((self.max_pose_detection, COCOBody.NUM_KEYPOINTS, 2))

        outputs = self.posenet(image)
        if self.model_type == "resnet":
            outputs[0] = tf.keras.activations.sigmoid(outputs[0])

        pose_count = decode_multiple_poses(
            outputs,
            keypoint_scores,
            keypoints,
            OUTPUT_STRIDE,
            min_pose_score=self.score_threshold,
        )

        keypoint_scores = keypoint_scores[:pose_count]
        keypoints = keypoints[:pose_count]

        # Convert coordinate to be relative to image size
        keypoints = keypoints * output_scale / image_size

        return keypoints, keypoint_scores

    def _preprocess(self, frame: np.ndarray) -> Tuple[tf.Tensor, List[int], List[int]]:
        """Rescale raw frame and convert to tensor image for inference"""
        image_size = [frame.shape[1], frame.shape[0]]
        target_height, target_width = self._get_valid_resolution(OUTPUT_STRIDE)
        scale = [frame.shape[1] / target_width, frame.shape[0] / target_height]

        scaled_image = cv2.resize(
            frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR
        )
        scaled_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB).astype(np.float32)

        if self.model_type == "resnet":
            scaled_image += IMAGE_NET_MEAN
        else:
            # Scale to [-1, 1] range
            scaled_image = scaled_image / 127.5 - 1.0

        scaled_image = np.expand_dims(scaled_image, axis=0)
        image_tensor = tf.convert_to_tensor(scaled_image)

        return image_tensor, scale, image_size
