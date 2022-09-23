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

import logging
from typing import Any, Dict, Set, Tuple, Union

import mediapipe as mp
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

from peekingduck.pipeline.nodes.base import ThresholdCheckerMixin

SUBTASK_MODEL_TYPES: Dict[str, Set[Union[float, int, str]]] = {"person": {0, 1, 2}}
KEYPOINT_33_TO_17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
# fmt: off
SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
            [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]]
# fmt: on


class PoseEstimationModel(ThresholdCheckerMixin):
    """MediaPipe pose estimation model class."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.check_valid_choice("subtask", {"person"})
        self.check_valid_choice(
            "model_type", SUBTASK_MODEL_TYPES[self.config["subtask"]]
        )
        self.check_bounds("score_threshold", "[0, 1]")

        self._create_mediapipe_model(
            self.config["subtask"],
            self.config["model_type"],
            self.config["score_threshold"],
        )

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Predicts bboxes from image.

        Args:
            image (np.ndarray): Input image frame.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains:
            - An array of detection bboxes
            - An array of detection labels
            - An array of detection scores

        Raises:
            TypeError: The provided `image` is not a numpy array.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a np.ndarray")

        result = self.model.process(image)
        return self._postprocess(result.pose_landmarks)

    def _create_mediapipe_model(
        self, subtask: str, model_type: int, score_threshold: float
    ) -> None:
        """Creates the MediaPipe model and logs the settings used."""
        if subtask == "person":
            self.model = mp.solutions.pose.Pose(
                model_complexity=model_type, min_detection_confidence=score_threshold
            )
        else:
            raise NotImplementedError(
                "Currently, only person pose estimation is implemented."
            )
        self.logger.info(
            "MediaPipe model loaded with the following configs:\n\t"
            f"Subtask: {subtask}\n\t"
            f"Model type: {model_type}\n\t"
            f"Score threshold: {score_threshold}"
        )

    def _postprocess(self, pose: NormalizedLandmarkList) -> Tuple[np.ndarray, ...]:
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
        landmarks = [pose.landmark[i] for i in KEYPOINT_33_TO_17]
        keypoints = np.array([[[landmark.x, landmark.y] for landmark in landmarks]])
        connections = np.array(
            [
                [
                    np.vstack((keypoints[0, start - 1], keypoints[0, stop - 1]))
                    for start, stop in SKELETON
                ]
            ]
        )
        scores = np.array([landmark.visibility] for landmark in landmarks)
        bboxes = np.array([self._get_bbox_from_pose(pose) for pose in keypoints])
        labels = np.array(["person"] * len(bboxes))
        return bboxes, labels, keypoints, connections, scores

    @staticmethod
    def _get_bbox_from_pose(pose: np.ndarray) -> np.ndarray:
        """Get the bounding box bordering the keypoints of a single pose"""
        if pose.shape[0]:
            min_x = pose[:, 0].min()
            min_y = pose[:, 1].min()
            max_x = pose[:, 0].max()
            max_y = pose[:, 1].max()
            return np.array([min_x, min_y, max_x, max_y])
        return np.empty((0, 4))
