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

"""Pose estimation subtask models."""

from typing import Any, Dict, Tuple

import mediapipe as mp
import numpy as np

from peekingduck.nodes.model.mediapipe_hubv1.api_doc import SUPPORTED_TASKS
from peekingduck.nodes.model.mediapipe_hubv1.subtask_model import base
from peekingduck.utils.pose.keypoint_handler import (
    BlazePoseBody,
    COCOBody,
    COCOHand,
    KeypointHandler,
)


class BodyEstimator(base.BaseEstimator):
    """Mediapipe pose estimation (body) model."""

    KEYPOINT_FORMATS = SUPPORTED_TASKS.get_keypoint_formats("body")

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        self.settings.append(
            base.ModelSetting(
                "smooth_landmarks", "Smooth landmarks", config["smooth_landmarks"]
            )
        )
        self.keypoint_handler = self._get_keypoint_handler(config["keypoint_format"])
        self.model = mp.solutions.pose.Pose(**self.arguments)

    def postprocess(self, result: Any) -> Tuple[np.ndarray, ...]:
        """Post processes detection result. Converts the bounding boxes from
        normalized [t, l, w, h] to normalized [x1, y1, x2, y2] format. Manually
        creates a "face" detection label for each detection.

        Args:
            result (Any): Pose estimation results which consists of landmark
                coordinates and visibility scores.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains:
            - An array of detection bboxes
            - An array of human-friendly detection class names
            - An array of keypoint coordinates
            - An array of keypoint connections
            - An array of keypoint scores
        """
        if result.pose_landmarks is None:
            return (
                np.empty((0, 4)),
                np.empty(0),
                np.empty(0),
                np.empty(0),
                np.empty(0),
            )

        pose = result.pose_landmarks
        raw_keypoints = [[[landmark.x, landmark.y] for landmark in pose.landmark]]
        raw_scores = [[landmark.visibility for landmark in pose.landmark]]

        self.keypoint_handler.update(raw_keypoints, raw_scores)
        labels = np.array(["person"] * len(self.keypoint_handler.scores))

        return (
            self.keypoint_handler.bboxes,
            labels,
            self.keypoint_handler.keypoints,
            self.keypoint_handler.connections,
            self.keypoint_handler.scores,
        )

    @staticmethod
    def _get_keypoint_handler(keypoint_format: str) -> KeypointHandler:
        if keypoint_format == "blaze_pose":
            return BlazePoseBody()
        return COCOBody([0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28])


class HandEstimator(base.BaseEstimator):
    """Mediapipe pose estimation (hand) model."""

    KEYPOINT_FORMATS = SUPPORTED_TASKS.get_keypoint_formats("hand")

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        self.settings.extend(
            (
                base.ModelSetting(
                    "max_num_hands", "Maximum number of hands", config["max_num_hands"]
                ),
                base.ModelSetting(None, "Mirror image", config["mirror_image"]),
            )
        )
        self.keypoint_handler = COCOHand()
        self.model = mp.solutions.hands.Hands(**self.arguments)

    def postprocess(self, result: Any) -> Tuple[np.ndarray, ...]:
        """Post processes detection result. Converts the bounding boxes from
        normalized [t, l, w, h] to normalized [x1, y1, x2, y2] format. Manually
        creates a "left hand" or "right hand" detection label for each
        detection.

        Args:
            result (Any): Pose estimation results which consists of landmark
                coordinates and visibility scores.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains:
            - An array of detection bboxes
            - An array of human-friendly detection class names
            - An array of keypoint coordinates
            - An array of keypoint connections
            - An array of keypoint scores
        """
        if result.multi_hand_landmarks is None:
            return (
                np.empty((0, 4)),
                np.empty(0),
                np.empty(0),
                np.empty(0),
                np.empty(0),
            )

        poses = result.multi_hand_landmarks
        raw_keypoints = [
            [[landmark.x, landmark.y] for landmark in pose.landmark] for pose in poses
        ]
        raw_scores = [
            [landmark.visibility for landmark in pose.landmark] for pose in poses
        ]
        self.keypoint_handler.update(raw_keypoints, raw_scores)
        labels = self.process_labels(result, self.config["mirror_image"])

        return (
            self.keypoint_handler.bboxes,
            labels,
            self.keypoint_handler.keypoints,
            self.keypoint_handler.connections,
            self.keypoint_handler.scores,
        )

    @staticmethod
    def process_labels(result: Any, mirror_image: bool) -> np.ndarray:
        """Processes the handedness classification results. Reverses the labels
        if the input image is not mirrored.

        Args:
            result (Any): Pose estimation results which consists of landmark
                coordinates and visibility scores.
            mirror_image (bool): Flag to indicate if input image is mirrored.

        Returns:
            (np.ndarray): An array of labels of left/right hands.
        """
        # Reverse the predicted direction if input image is not mirrored
        reversed_direction = {"left": "right", "right": "left"}
        labels = []
        for handedness in result.multi_handedness:
            direction = handedness.classification[0].label.lower()
            if not mirror_image:
                direction = reversed_direction[direction]
            labels.append(f"{direction} hand")
        return np.array(labels)
