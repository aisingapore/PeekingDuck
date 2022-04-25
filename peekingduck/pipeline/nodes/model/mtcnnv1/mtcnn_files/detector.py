# Copyright 2022 AI Singapore
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

"""Face detection class using MTCNN model to find face bboxes."""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from peekingduck.pipeline.nodes.model.mtcnnv1.mtcnn_files.graph_functions import (
    load_graph,
)
from peekingduck.pipeline.utils.bbox.transforms import xyxy2xyxyn


class Detector:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """Face detection class using MTCNN model to find bboxes and landmarks."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model_dir: Path,
        model_type: str,
        model_file: Dict[str, str],
        model_nodes: Dict[str, List[str]],
        min_size: int,
        scale_factor: float,
        network_thresholds: List[float],
        score_threshold: float,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        self.model_path = model_dir / model_file[model_type]
        self.model_nodes = model_nodes

        self.min_size = min_size
        self.scale_factor = scale_factor
        self.network_thresholds = network_thresholds
        self.score_threshold = score_threshold

        self.mtcnn = self._create_mtcnn_model()

    def predict_object_bbox_from_image(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts face bboxes, scores, and landmarks.

        Args:
            image (np.ndarray): Image in numpy array

        Returns:
            bboxes (np.ndarray): Detected bboxes.
            scores (np.ndarray): Confidence scores.
            landmarks (np.ndarray): Facial landmarks.
        """
        image = self._preprocess(image)
        bboxes, scores, landmarks = self.mtcnn(
            image, self.min_size, self.scale_factor, self.network_thresholds
        )
        bboxes, scores, landmarks = self._post_process(image, bboxes, scores, landmarks)

        return bboxes, scores, landmarks

    def _create_mtcnn_model(self) -> Callable:
        """Creates MTCNN model for face detection."""
        self.logger.info(
            "MTCNN model loaded with following configs:\n\t"
            f"Min size: {self.min_size},\n\t"
            f"Scale Factor: {self.scale_factor},\n\t"
            f"Network Thresholds: {self.network_thresholds},\n\t"
            f"Score Threshold: {self.score_threshold}"
        )

        return self._load_mtcnn_weights()

    def _load_mtcnn_weights(self) -> Callable:
        if not self.model_path.is_file():
            raise ValueError(
                f"Graph file does not exist. Please check that {self.model_path} exists"
            )

        return load_graph(
            str(self.model_path),
            self.model_nodes["inputs"],
            self.model_nodes["outputs"],
        )

    def _post_process(
        self,
        image: np.ndarray,
        bboxes: tf.Tensor,
        scores: tf.Tensor,
        landmarks: tf.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Processes MTCNN model outputs. Filter detections by confidence score
        and swaps the x and y coordinates of the bboxes.

        Args:
            image (np.ndarray): Image in numpy array.
            bboxes (tf.Tensor): Tensor array of detected bboxes.
            scores (tf.Tensor): Tensor array of confidence scores.
            landmarks (tf.Tensor): Tensor array of facial landmarks.

        Returns:
            bboxes (np.ndarray): Processed detected bboxes.
            scores (np.ndarray): Processed confidence scores.
            landmarks (np.ndarray): Processed facial landmarks.
        """
        bboxes, scores, landmarks = bboxes.numpy(), scores.numpy(), landmarks.numpy()

        # Filter bboxes by confidence score
        indices = np.where(scores > self.score_threshold)[0]
        bboxes = bboxes[indices]
        scores = scores[indices]
        landmarks = landmarks[indices]

        # Swap position of x, y coordinates
        bboxes[:, [0, 1]] = bboxes[:, [1, 0]]
        bboxes[:, [2, 3]] = bboxes[:, [3, 2]]

        bboxes = xyxy2xyxyn(bboxes, image.shape[0], image.shape[1])

        return bboxes, scores, landmarks

    @staticmethod
    def _preprocess(image: np.ndarray) -> tf.Tensor:
        """Processes input image

        Args:
            image (np.ndarray): image in numpy array

        Returns:
            image (np.ndarray): processed numpy array of image
        """
        image = image.astype(np.float32)
        image = tf.convert_to_tensor(image)

        return image
