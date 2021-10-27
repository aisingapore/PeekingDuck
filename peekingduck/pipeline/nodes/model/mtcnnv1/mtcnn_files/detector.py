# Copyright 2021 AI Singapore

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
Face detection class using mtcnn model to find face bboxes
"""

import os
import logging
from typing import Dict, Any, Tuple, List

import numpy as np
import tensorflow as tf

from peekingduck.pipeline.nodes.model.mtcnnv1.mtcnn_files.graph_functions import (
    load_graph,
)


class Detector:  # pylint: disable=too-many-instance-attributes
    """Face detection class using MTCNN model to find bboxes and landmarks"""

    def __init__(self, config: Dict[str, Any]) -> None:

        self.logger = logging.getLogger(__name__)

        self.config = config
        self.root_dir = config["root"]
        self.min_size = self.config["mtcnn_min_size"]
        self.factor = self.config["mtcnn_factor"]
        self.thresholds = self.config["mtcnn_thresholds"]
        self.score = self.config["mtcnn_score"]
        self.mtcnn = self._create_mtcnn_model()

    def _create_mtcnn_model(self) -> tf.keras.Model:
        """
        Creates MTCNN model for face detection
        """
        model_path = os.path.join(self.root_dir, self.config["graph_files"]["mtcnn"])

        self.logger.info(
            "MTCNN model loaded with following configs: \n \
            Min size: %s, \n \
            Scale Factor: %s, \n \
            Steps Thresholds: %s, \n \
            Score Threshold: %s",
            self.config["mtcnn_min_size"],
            self.config["mtcnn_factor"],
            self.config["mtcnn_thresholds"],
            self.config["mtcnn_score"],
        )

        return self._load_mtcnn_graph(model_path)

    @staticmethod
    def _load_mtcnn_graph(filepath: str) -> tf.compat.v1.GraphDef:
        model_path = os.path.join(filepath)
        if os.path.isfile(model_path):
            return load_graph(model_path)

        raise ValueError(
            "Graph file does not exist. Please check that " "%s exists" % model_path
        )

    def predict_bbox_landmarks(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predicts face bboxes, scores and landmarks

        Args:
            image (np.ndarray): image in numpy array

        Returns:
            bboxes (np.ndarray): numpy array of detected bboxes
            scores (np.ndarray): numpy array of confidence scores
            landmarks (np.ndarray): numpy array of facial landmarks
            labels (np.ndarray): numpy array of class labels (i.e. face)
        """
        # 1. process inputs
        image = self.process_image(image)
        min_size, factor, thresholds = self.process_params(
            self.min_size, self.factor, self.thresholds
        )

        # 2. evaluate image
        bboxes, scores, landmarks = self.mtcnn(image, min_size, factor, thresholds)

        # 3. process outputs
        bboxes, scores, landmarks = self.process_outputs(
            image, bboxes, scores, landmarks
        )

        # 4. create bbox_labels
        classes = np.array(["face"] * len(bboxes))

        return bboxes, scores, landmarks, classes

    @staticmethod
    def process_image(image: np.ndarray) -> tf.Tensor:
        """Processes input image

        Args:
            image (np.ndarray): image in numpy array

        Returns:
            image (np.ndarray): processed numpy array of image
        """
        image = image.astype(np.float32)
        image = tf.convert_to_tensor(image)

        return image

    @staticmethod
    def process_params(
        min_size: int, factor: float, thresholds: List[float]
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Processes model input parameters

        Args:
            min_size (int): minimum face size
            factor (float): scale factor
            thresholds (list): steps thresholds

        Returns:
            min_size (tf.Tensor): processed minimum face size
            factor (tf.Tensor): processed scale factor
            thresholds (tf.Tensor): processed steps thresholds
        """
        min_size = tf.convert_to_tensor(float(min_size))
        factor = tf.convert_to_tensor(float(factor))
        thresholds = [float(integer) for integer in thresholds]
        thresholds = tf.convert_to_tensor(thresholds)

        return min_size, factor, thresholds

    def process_outputs(
        self,
        image: np.ndarray,
        bboxes: tf.Tensor,
        scores: tf.Tensor,
        landmarks: tf.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Processes MTCNN model outputs

        Args:
            image (np.ndarray): image in numpy array
            bboxes (tf.Tensor): tensor array of detected bboxes
            scores (tf.Tensor): tensor array of confidence scores
            landmarks (tf.Tensor): tensor array of facial landmarks

        Returns:
            bboxes (np.ndarray): processed numpy array of detected bboxes
            scores (np.ndarray): processed numpy array of confidence scores
            landmarks (np.ndarray): processed numpy array of facial landmarks
        """
        bboxes, scores, landmarks = bboxes.numpy(), scores.numpy(), landmarks.numpy()

        # Filter bboxes by confidence score
        indices = np.where(scores > self.score)[0]
        bboxes = bboxes[indices]
        scores = scores[indices]
        landmarks = landmarks[indices]

        # Swap position of x, y coordinates
        bboxes[:, [0, 1]] = bboxes[:, [1, 0]]
        bboxes[:, [2, 3]] = bboxes[:, [3, 2]]

        # Express image coordinates as a percentage of image height and width
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / image.shape[1]
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / image.shape[0]

        return bboxes, scores, landmarks
