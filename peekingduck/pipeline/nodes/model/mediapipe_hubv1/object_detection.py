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
from peekingduck.pipeline.utils.bbox.transforms import tlwhn2xyxyn


class ObjectDetectionModel(MediaPipeModel):
    """MediaPipe object detection model class."""

    TASK = "object_detection"
    SUBTASK_MODEL_TYPES = SUPPORTED_TASKS.get_subtask_model_types(TASK)
    SUBTASKS = SUPPORTED_TASKS.get_subtasks(TASK)

    @property
    def model_settings(self) -> Dict[str, Any]:
        """Constructs the model configuration options based on subtask type."""
        if self.subtask == "face":
            return {
                "min_detection_confidence": self.config["score_threshold"],
                "model_selection": self.config["model_type"],
            }
        raise NotImplementedError("Only `face` detection is implemented.")

    def _create_mediapipe_model(self, model_settings: Dict[str, Any]) -> None:
        """Creates the MediaPipe model and logs the settings used."""
        if self.subtask == "face":
            self.model = mp.solutions.face_detection.FaceDetection(**model_settings)
            self.logger.info(
                "MediaPipe model loaded with the following configs:\n\t"
                f"Subtask: {self.subtask}\n\t"
                f"Model type: {model_settings['model_selection']}\n\t"
                f"Score threshold: {model_settings['min_detection_confidence']}"
            )
        else:
            raise NotImplementedError("Only `face` detection is implemented.")

    def _postprocess(self, result: Any) -> Tuple[np.ndarray, ...]:
        """Post processes detection result. Converts the bounding boxes from
        normalized [t, l, w, h] to normalized [x1, y1, x2, y2] format. Creates
        "face" detection label for each detection.

        Args:
            detections (Optional[Detection]): Detection results which consist
                of bounding boxes and confidence scores.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains:
            - An array of detection bboxes
            - An array of human-friendly detection class names
            - An array of detection scores
        """
        if result.detections is None:
            return np.empty((0, 4)), np.empty(0), np.empty(0)

        detections = result.detections
        bboxes = tlwhn2xyxyn(
            np.array(
                [
                    [
                        detection.location_data.relative_bounding_box.xmin,
                        detection.location_data.relative_bounding_box.ymin,
                        detection.location_data.relative_bounding_box.width,
                        detection.location_data.relative_bounding_box.height,
                    ]
                    for detection in detections
                ]
            )
        )
        labels = np.array(["face"] * len(bboxes))
        scores = np.array([detection.score for detection in detections])
        return bboxes, labels, scores
