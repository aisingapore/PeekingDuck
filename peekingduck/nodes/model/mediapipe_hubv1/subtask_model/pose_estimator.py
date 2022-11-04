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

        self.keypoint_handler = self._get_keypoint_handler(config["keypoint_format"])
        self.settings.append(
            base.ModelSetting(
                "smooth_landmarks", "Smooth landmarks", config["smooth_landmarks"]
            )
        )
        self.model = mp.solutions.pose.Pose(**self.arguments)

    def postprocess(self, result: Any) -> Tuple[np.ndarray, ...]:
        """Post processes detection result. Converts the bounding boxes from
        normalized [t, l, w, h] to normalized [x1, y1, x2, y2] format. Manually
        creates a "face" detection label for each
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

        self.keypoint_handler.update_keypoints(raw_keypoints)
        scores = self.keypoint_handler.convert_scores(raw_scores)
        labels = np.array(["person"] * len(scores))

        return (
            self.keypoint_handler.bboxes,
            labels,
            self.keypoint_handler.keypoints,
            self.keypoint_handler.connections,
            scores,
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

        self.keypoint_handler = COCOHand()
        self.settings.append(
            base.ModelSetting(
                "max_num_hands", "Maximum number of hands", config["max_num_hands"]
            )
        )
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
        keypoints_33 = [
            [[landmark.x, landmark.y] for landmark in pose.landmark] for pose in poses
        ]
        scores_33 = [
            [landmark.visibility for landmark in pose.landmark] for pose in poses
        ]
        self.keypoint_handler.update_keypoints(keypoints_33)
        scores = self.keypoint_handler.convert_scores(scores_33)
        labels = np.array(["hand"] * len(scores))

        return (
            self.keypoint_handler.bboxes,
            labels,
            self.keypoint_handler.keypoints,
            self.keypoint_handler.connections,
            scores,
        )
