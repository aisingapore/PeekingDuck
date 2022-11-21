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

import numpy as np
import tensorflow as tf

from peekingduck.nodes.model.posenetv1.posenet_files.constants import (
    KEYPOINTS_NUM,
    MIN_PART_SCORE,
    SCALE_FACTOR,
)
from peekingduck.nodes.model.posenetv1.posenet_files.detector import (
    detect_keypoints,
    get_keypoints_relative_coords,
)
from peekingduck.nodes.model.posenetv1.posenet_files.preprocessing import rescale_image
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
        full_keypoint_rel_coords, full_keypoint_scores = self._predict_all_poses(frame)
        self.keypoint_handler.update(full_keypoint_rel_coords, full_keypoint_scores)

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
        image, output_scale, image_size = self._preprocess(frame)

        dst_scores = np.zeros((self.max_pose_detection, KEYPOINTS_NUM))
        dst_keypoints = np.zeros((self.max_pose_detection, KEYPOINTS_NUM, 2))

        pose_count = detect_keypoints(
            self.posenet,
            image,
            OUTPUT_STRIDE,
            dst_scores,
            dst_keypoints,
            self.model_type,
            self.score_threshold,
        )
        full_keypoint_scores = dst_scores[:pose_count]
        full_keypoint_coords = dst_keypoints[:pose_count]

        full_keypoint_rel_coords = get_keypoints_relative_coords(
            full_keypoint_coords, output_scale, image_size
        )

        return full_keypoint_rel_coords, full_keypoint_scores

    def _preprocess(self, frame: np.ndarray) -> Tuple[tf.Tensor, np.ndarray, List[int]]:
        """Rescale raw frame and convert to tensor image for inference"""
        image_size = [frame.shape[1], frame.shape[0]]

        image, output_scale = rescale_image(
            frame,
            self.resolution,
            SCALE_FACTOR,
            OUTPUT_STRIDE,
            self.model_type,
        )
        image = tf.convert_to_tensor(image)
        return image, output_scale, image_size
