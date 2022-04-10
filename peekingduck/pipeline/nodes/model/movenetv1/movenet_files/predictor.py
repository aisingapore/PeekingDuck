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
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

# fmt: off
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
    [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
    [4, 6], [5, 7],
]
# fmt: on


class Predictor:  # pylint: disable=too-many-instance-attributes
    """Predictor class to handle detection of poses for MoveNet."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model_dir: Path,
        model_type: str,
        model_file: Dict[str, str],
        resolution: Dict[str, Dict[str, int]],
        bbox_score_threshold: float,
        keypoint_score_threshold: float,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        self.model_type = model_type
        self.model_path = model_dir / model_file[self.model_type]
        self.resolution = self.get_resolution_as_tuple(resolution[self.model_type])

        self.bbox_score_threshold = bbox_score_threshold
        self.keypoint_score_threshold = keypoint_score_threshold

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
            f"Model type: {self.model_type},\n\t"
            f"Input resolution: {self.resolution},\n\t"
            f"bbox_score_threshold: {bbox_score_threshold},\n\t"
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
        keypoints_x = predictions[:, range(1, 51, 3)]
        keypoints_y = predictions[:, range(0, 51, 3)]
        keypoints_scores = predictions[:, range(2, 51, 3)]
        bboxes = predictions[:, 51:55]
        bbox_score = predictions[:, 55]
        keypoints = (
            tf.concat(
                [
                    tf.expand_dims(keypoints_x, axis=2),
                    tf.expand_dims(keypoints_y, axis=2),
                ],
                2,
            )
        ).numpy()
        # swap bbox coordinates from y1,x1,y2,x2 to x1,y1,x2,y2
        bboxes[:, [0, 1]] = bboxes[:, [1, 0]]
        bboxes[:, [2, 3]] = bboxes[:, [3, 2]]
        bbox_masks = self._get_masks_from_bbox_scores(
            bbox_score, self.bbox_score_threshold
        )
        if len(bbox_masks) == 0:
            return (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))

        bboxes = bboxes[bbox_masks, :]
        bbox_score = bbox_score[bbox_masks]
        keypoints = keypoints[bbox_masks, :, :]
        keypoints_scores = keypoints_scores[bbox_masks, :]
        valid_keypoints, keypoints_masks = self._get_keypoints_coords(
            keypoints, keypoints_scores, self.keypoint_score_threshold
        )
        keypoints_conns = self._get_connections_of_poses(keypoints, keypoints_masks)

        return bboxes, valid_keypoints, keypoints_scores, keypoints_conns

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
        predictions = tf.squeeze(predictions, axis=0)
        predictions = tf.squeeze(predictions, axis=0).numpy()
        keypoints_x = predictions[:, 1]
        keypoints_y = predictions[:, 0]
        keypoints_scores = predictions[:, 2].reshape((1, -1))

        # to concatenate the xy coordinate together for each keypoint
        keypoints = (
            tf.concat(
                [
                    tf.expand_dims(keypoints_x, axis=1),
                    tf.expand_dims(keypoints_y, axis=1),
                ],
                1,
            )
        ).numpy()
        keypoints = keypoints.reshape((1, -1, 2))
        valid_keypoints, keypoints_masks = self._get_keypoints_coords(
            keypoints, keypoints_scores, self.keypoint_score_threshold
        )
        # if valid_keypoints is None:
        #     return (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
        # if all the keypoints_score are below threshold, valid_keypoints
        # will be an array with all [-1,-1]
        if np.all(valid_keypoints == np.asarray([[-1, -1]])):
            return (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))

        keypoints_conns = self._get_connections_of_poses(keypoints, keypoints_masks)
        bbox = self._get_bbox_from_keypoints(valid_keypoints)

        return bbox, valid_keypoints, keypoints_scores, keypoints_conns

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

    @staticmethod
    def _get_bbox_from_keypoints(valid_keypoints: np.ndarray) -> np.ndarray:
        """Obtains coordinates from the keypoints for single pose model where
        bounding box coordinates are not provided as model outputs. The bounding
        box coordinates will be [xmin,ymin,xmax,ymax], derived from the keypoints
        coordinates.

        Args:
            valid_keypoints (np.ndarray): Nx17x2 array of keypoints where N is
                the number of detections.

        Returns:
            bboxes (np.ndarray): Nx4 array of bbox coordinates in the form of
                [x1,y1,x2,y2] where the values are in relative coordinates
                between 0 and 1.
        """
        # using absolute because it is coded that invalid keypoints are set as -1
        xmin = np.abs(valid_keypoints[:, :, 0]).min()
        ymin = np.abs(valid_keypoints[:, :, 1]).min()
        xmax = valid_keypoints[:, :, 0].max()
        ymax = valid_keypoints[:, :, 1].max()
        bboxes = np.asarray([[xmin, ymin, xmax, ymax]])

        return bboxes

    @staticmethod
    def _get_connections_of_poses(
        keypoints: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        """Gets connections between adjacent keypoint pairs if both keypoints are
        detected. Output will be in the shape of NxD'x2 keypoint connections,
        where D' is the varying pairs of valid keypoint connections per
        detection.

        Args:
            keypoints (np.ndarray): Nx17x2 array of keypoint coordinates, where N
                is the number of detections.
            masks (np.ndarray): Nx17 boolean for valid keypoints.

        Returns:
            keypoint_conns (np.ndarray): NxD'x2 keypoint connections, where D' is
                the varying pair of valid keypoint connections per detection
        """
        all_connections = []
        for i in range(keypoints.shape[0]):
            connections = []
            for start_joint, end_joint in SKELETON:
                if masks[i, start_joint - 1] and masks[i, end_joint - 1]:
                    connections.append(
                        np.vstack(
                            (keypoints[i, start_joint - 1], keypoints[i, end_joint - 1])
                        )
                    )
            all_connections.append(connections)
        keypoint_conns = np.asarray(all_connections)

        return keypoint_conns

    @staticmethod
    def _get_keypoints_coords(
        keypoints: np.ndarray,
        keypoints_score: np.ndarray,
        keypoints_score_threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Keeps only valid keypoints' relative coordinates.

        Args:
            keypoints (np.ndarray): Nx17x2 array of keypoints' relative
                coordinates.
            keypoints_score (np.ndarray): Nx17 keypoints confidence score for
                valid keypoints coordinates.
            keypoints_score_threshold (float): threshold for keypoints score.

        Returns:
            valid_keypoints (np.ndarray): Nx17x2 array of keypoints where
                undetected keypoints are assigned a (-1,-1) value.
            keypoint_masks (np.ndarray) : Nx17 boolean for valid keypoints
        """
        valid_keypoints = keypoints.copy()
        valid_keypoints = np.clip(valid_keypoints, 0, 1)
        keypoint_masks = keypoints_score > keypoints_score_threshold
        valid_keypoints[~keypoint_masks] = [-1, -1]

        return valid_keypoints, keypoint_masks

    @staticmethod
    def _get_masks_from_bbox_scores(
        bbox_scores: np.ndarray, score_threshold: float
    ) -> List[int]:
        """Obtains masks for bbox with confidence scores above the detection
        threshold. The mask come in the format of a list with idx of approved
        detection, e.g., [0,1,2], generally movenet will have the valid
        detections in the first few indexes

        Args:
            bbox_scores (np.ndarray): Array of bbox confidence scores.
            score_threshold (float): Threshold value for bbox confidence scores.

        Returns:
            masks (list): List of index of bboxes that have respective bbox
                confidence scores above the score threshold.
        """
        bbox_scores_list = bbox_scores.tolist()
        masks = [
            idx for idx, score in enumerate(bbox_scores_list) if score > score_threshold
        ]
        return masks
