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
Predictor class to handle detection of poses for movenet
"""

import logging
from pathlib import Path
from typing import Callable, Dict, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from peekingduck.utils.pose.keypoint_handler import COCOBody


class Predictor:  # pylint: disable=too-many-instance-attributes
    """Predictor class to handle detection of poses for MoveNet."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model_dir: Path,
        model_format: str,
        model_type: str,
        model_file: Dict[str, str],
        resolution: Dict[str, Dict[str, int]],
        bbox_score_threshold: float,
        keypoint_score_threshold: float,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        self.model_format = model_format
        self.model_type = model_type
        self.model_path = model_dir / model_file[self.model_type]
        self.resolution = self.get_resolution_as_tuple(resolution[self.model_type])

        self.bbox_score_threshold = bbox_score_threshold
        self.keypoint_score_threshold = keypoint_score_threshold

        self.keypoint_handler = COCOBody(score_threshold=keypoint_score_threshold)
        self.movenet = self._create_movenet_model()

    def predict(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # pylint: disable=too-many-locals
        """MoveNet prediction function

        Args:
            frame (np.ndarray): image in numpy array

        Returns:
            bboxes (np.ndarray): Nx4 array of bboxes, N is number of detections
            keypoints (np.ndarray): Nx17x2 array of keypoint coordinates
            keypoints_scores (np.ndarray): Nx17 array of keypoint scores
            keypoints_conns (np.ndarray): NxD'x2 keypoint connections, where
                D' is the varying pairs of valid keypoint connections per detection
        """
        image_data = cv2.resize(frame, (self.resolution))
        image_data = np.asarray([image_data]).astype(np.int32)
        outputs = self.movenet(tf.constant(image_data))
        predictions = outputs["output_0"]

        if "multi" in self.model_type:
            (
                bboxes,
                keypoints,
                keypoints_scores,
                keypoints_conns,
            ) = self._get_results_multi(predictions)
        else:
            (
                bboxes,
                keypoints,
                keypoints_scores,
                keypoints_conns,
            ) = self._get_results_single(predictions)

        return bboxes, keypoints, keypoints_scores, keypoints_conns

    def _create_movenet_model(self) -> Callable:
        """Creates the MoveNet model."""
        # movenet singlepose do not output bbox, so bbox score threshold not
        # applicable
        bbox_score_threshold = (
            self.bbox_score_threshold
            if "multi" in self.model_type
            else "NA for singlepose models"
        )
        self.logger.info(
            f"MoveNet model loaded with following configs:\n\t"
            f"Model format: {self.model_format}\n\t"
            f"Model type: {self.model_type}\n\t"
            f"Input resolution: {self.resolution}\n\t"
            f"bbox_score_threshold: {bbox_score_threshold}\n\t"
            f"keypoint_score_threshold: {self.keypoint_score_threshold}"
        )

        return self._load_movenet_weights()

    def _get_results_multi(
        self, predictions: tf.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns formatted outputs for multipose model.

        Predictions come in a [1x6x56] tensor for up to 6 people detections.
        First 17 x 3 = 51 elements are keypoints locations in the form of
        [y_0, x_0, s_0, y_1, x_1, s_1, …, y_16, x_16, s_16],
        where y_i, x_i, s_i are the yx-coordinates and confidence score.
        Remaining 5 elements [ymin, xmin, ymax, xmax, score] represent
        bbox coordinates, and confidence score

        Args:
            predictions (tf.Tensor): Model output in a [1x6x56] tensor.

        Returns:
            bboxes (np.ndarray): Nx4 array of bboxes, N is number of detections.
            keypoints (np.ndarray): Nx17x2 array of keypoint coordinates.
            keypoints_scores (np.ndarray): Nx17 array of keypoint scores.
            keypoints_conns (np.ndarray): NxD'x2 keypoint connections, where
                D' is the varying pair of valid keypoint connections per
                detection.
        """
        predictions = tf.squeeze(predictions, axis=0).numpy()

        bbox_scores = predictions[:, 55]
        conf_mask = bbox_scores >= self.bbox_score_threshold
        predictions = predictions[conf_mask]
        if predictions.shape[0] == 0:
            return np.empty((0, 4)), np.empty(0), np.empty(0), np.empty(0)

        keypoints_x = predictions[:, 1:51:3]
        keypoints_y = predictions[:, 0:51:3]
        keypoints_scores = predictions[:, 2:51:3]
        # swap bbox coordinates from y1,x1,y2,x2 to x1,y1,x2,y2
        bboxes = predictions[:, [52, 51, 54, 53]]

        keypoints = np.stack([keypoints_x, keypoints_y], axis=2)
        self.keypoint_handler.update(keypoints, keypoints_scores)

        return (
            bboxes,
            self.keypoint_handler.keypoints,
            self.keypoint_handler.scores,
            self.keypoint_handler.connections,
        )

    def _get_results_single(
        self, predictions: tf.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns formatted outputs for singlepose model.

        Predictions come in a [1x1x17x3] tensor.
        First 2 channel in the last dimension represent the (y,x) coordinates,
        the 3rd channel represents confidence scores for the keypoints.

        Args:
            predictions (tf.Tensor): Model output in a [1x1x17x3] tensor.

        Returns:
            bboxes (np.ndarray): 1x4 array of bboxes.
            keypoints (np.ndarray): 1x17x2 array of keypoint coordinates.
            keypoints_scores (np.ndarray): 1x17 array of keypoint scores.
            keypoints_conns (np.ndarray): 1xD'x2 keypoint connections, where
                D' is the varying pair of valid keypoint connections per
                detection.
        """
        predictions = tf.squeeze(predictions, axis=0).numpy()

        # Swap y,x to x,y
        keypoints = predictions[..., [1, 0]]
        keypoints_scores = predictions[..., 2]

        self.keypoint_handler.update(keypoints, keypoints_scores)

        if not self.keypoint_handler.keypoint_masks.any():
            return np.empty((0, 4)), np.empty(0), np.empty(0), np.empty(0)

        return (
            self.keypoint_handler.bboxes,
            self.keypoint_handler.keypoints,
            self.keypoint_handler.scores,
            self.keypoint_handler.connections,
        )

    def _load_movenet_weights(self) -> Callable:
        self.model = tf.saved_model.load(
            str(self.model_path), tags=[tag_constants.SERVING]
        )
        return self.model.signatures["serving_default"]

    @staticmethod
    def get_resolution_as_tuple(resolution: Dict[str, int]) -> Tuple[int, int]:
        """Convert resolution from dict to tuple format

        Args:
            resolution (Dict[str, int]): height and width in dict format

        Returns:
            resolution (Tuple(int)): height and width in tuple format
        """
        return int(resolution["height"]), int(resolution["width"])
