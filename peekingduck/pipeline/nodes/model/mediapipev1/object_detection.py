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
from typing import Any, Dict, Optional, Set, Tuple, Union

import mediapipe as mp
import numpy as np
from mediapipe.framework.formats.detection_pb2 import Detection

from peekingduck.pipeline.nodes.base import ThresholdCheckerMixin
from peekingduck.pipeline.utils.bbox.transforms import tlwhn2xyxyn

SUBTASK_MODEL_TYPES: Dict[str, Set[Union[float, int, str]]] = {"face": {0, 1}}


class ObjectDetectionModel(ThresholdCheckerMixin):
    """MediaPipe object detection model class."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.check_valid_choice("subtask", {"face"})
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
        return self._postprocess(result.detections)

    def _create_mediapipe_model(
        self, subtask: str, model_type: int, score_threshold: float
    ) -> None:
        """Creates the MediaPipe model and logs the settings used."""
        if subtask == "face":
            self.model = mp.solutions.face_detection.FaceDetection(
                model_selection=model_type, min_detection_confidence=score_threshold
            )
        else:
            raise NotImplementedError("Only `face` detection is implemented.")
        self.logger.info(
            "MediaPipe model loaded with the following configs:\n\t"
            f"Subtask: {subtask}\n\t"
            f"Model type: {model_type}\n\t"
            f"Score threshold: {score_threshold}"
        )

    @staticmethod
    def _postprocess(detections: Optional[Detection]) -> Tuple[np.ndarray, ...]:
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
        if detections is None:
            return np.empty((0, 4)), np.empty(0), np.empty(0)
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
