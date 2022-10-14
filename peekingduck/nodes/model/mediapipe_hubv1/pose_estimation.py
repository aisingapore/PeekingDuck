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

"""MediaPipe object detection model."""

from typing import Any, Dict, Tuple

import mediapipe as mp
import numpy as np

from peekingduck.pipeline.nodes.model.mediapipe_hubv1.api_doc import SUPPORTED_TASKS
from peekingduck.pipeline.nodes.model.mediapipe_hubv1.base import MediaPipeModel
from peekingduck.pipeline.utils.pose.coco import BodyKeypoint

KEYPOINT_33_TO_17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


class PoseEstimationModel(MediaPipeModel):
    """MediaPipe pose estimation model class."""

    TASK = "pose_estimation"
    SUBTASK_MODEL_TYPES = SUPPORTED_TASKS.get_subtask_model_types(TASK)
    SUBTASKS = SUPPORTED_TASKS.get_subtasks(TASK)

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.check_bounds("tracking_score_threshold", "[0, 1]")

        self.keypoint_handler = BodyKeypoint(KEYPOINT_33_TO_17)

    @property
    def model_settings(self) -> Dict[str, Any]:
        """Constructs the model configuration options based on subtask type."""
        if self.subtask == "body":
            return {
                "min_detection_confidence": self.config["score_threshold"],
                "min_tracking_confidence": self.config["tracking_score_threshold"],
                "model_complexity": self.config["model_type"],
                "smooth_landmarks": self.config["smooth_landmarks"],
                "static_image_mode": self.config["static_image_mode"],
            }
        raise NotImplementedError("Only `body` pose estimation is implemented.")

    def _create_mediapipe_model(self, model_settings: Dict[str, Any]) -> None:
        """Creates the MediaPipe model and logs the settings used."""
        if self.subtask == "body":
            self.model = mp.solutions.pose.Pose(**model_settings)
            self.logger.info(
                "MediaPipe model loaded with the following configs:\n\t"
                f"Subtask: {self.subtask}\n\t"
                f"Model type: {model_settings['model_complexity']}\n\t"
                f"Score threshold: {model_settings['min_detection_confidence']}\n\t"
                f"Tracking score threshold: {model_settings['min_tracking_confidence']}\n\t"
                f"Static image mode: {model_settings['static_image_mode']}\n\t"
                f"Smooth landmarks: {model_settings['smooth_landmarks']}"
            )
        else:
            raise NotImplementedError("Only `body` pose estimation is implemented.")

    def _postprocess(self, result: Any) -> Tuple[np.ndarray, ...]:
        """Post processes detection result. Converts the bounding boxes from
        normalized [t, l, w, h] to normalized [x1, y1, x2, y2] format. Creates
        "face" detection label for each detection.

        Args:
            result (Dict[str, torch.Tensor]): A dictionary containing the model
                output, with the keys: "boxes", "labels", "masks, and "scores".
            image_shape (Tuple[int, int]): The height and width of the input
                image.

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
            return np.empty((0, 4)), np.empty(0), np.empty(0), np.empty(0), np.empty(0)

        pose = result.pose_landmarks
        keypoints_33 = [[[landmark.x, landmark.y] for landmark in pose.landmark]]
        scores_33 = [[landmark.visibility for landmark in pose.landmark]]

        self.keypoint_handler.update_keypoints(keypoints_33)
        scores = self.keypoint_handler.convert_scores(scores_33)
        labels = np.array(["person"] * len(scores))

        return (
            self.keypoint_handler.bboxes,
            labels,
            self.keypoint_handler.keypoints,
            self.keypoint_handler.connections,
            scores,
        )
