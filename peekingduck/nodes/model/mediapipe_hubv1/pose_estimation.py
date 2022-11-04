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

"""MediaPipe pose estimation model."""

from typing import Any, Dict, Tuple

import numpy as np

from peekingduck.nodes.model.mediapipe_hubv1.api_doc import SUPPORTED_TASKS
from peekingduck.nodes.model.mediapipe_hubv1.base import MediaPipeModel
from peekingduck.nodes.model.mediapipe_hubv1.subtask_model import (
    BodyEstimator,
    HandEstimator,
)
from peekingduck.nodes.model.mediapipe_hubv1.subtask_model.base import BaseEstimator


class PoseEstimationModel(MediaPipeModel):
    """MediaPipe pose estimation model class."""

    TASK = "pose_estimation"
    SUBTASK_MODEL_TYPES = SUPPORTED_TASKS.get_subtask_model_types(TASK)
    SUBTASKS = SUPPORTED_TASKS.get_subtasks(TASK)

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.check_bounds("max_num_hands", "[0, +inf)")
        self.check_bounds("tracking_score_threshold", "[0, 1]")

        self.model = self._get_subtask_model(config)

    @property
    def model_settings(self) -> Dict[str, Any]:
        """Constructs the model configuration options based on subtask type."""
        return self.model.settings

    def _create_mediapipe_model(self, model_settings: Dict[str, Any]) -> None:
        """Creates the MediaPipe model and logs the settings used."""
        self.logger.info(
            "MediaPipe model loaded with the following configs:\n\t"
            f"Subtask: {self.subtask}\n\t"
            f"{self.model.loaded_config}"
        )

    def _get_subtask_model(self, config: Dict[str, Any]) -> BaseEstimator:
        if self.subtask == "body":
            return BodyEstimator(config)
        if self.subtask == "hand":
            return HandEstimator(config)
        raise NotImplementedError(
            f"Pose estimation subtask '{self.subtask}' is not implemented."
        )

    def _postprocess(self, result: Any) -> Tuple[np.ndarray, ...]:
        """Post processes detection result. Converts the bounding boxes from
        normalized [t, l, w, h] to normalized [x1, y1, x2, y2] format. Manually
        creates a detection label for each detection.

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
        return self.model.postprocess(result)
