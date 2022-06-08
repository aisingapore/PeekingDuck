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

"""YOLO-based face detection model with model types: v4 and v4tiny."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from peekingduck.pipeline.nodes.base import (
    ThresholdCheckerMixin,
    WeightsDownloaderMixin,
)
from peekingduck.pipeline.nodes.model.yolov4_face.yolo_face_files.detector import (
    Detector,
)


class YOLOFaceModel(ThresholdCheckerMixin, WeightsDownloaderMixin):
    """YOLO face model with model types: v4 and v4tiny."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.check_bounds(["iou_threshold", "score_threshold"], "[0, 1]")

        model_dir = self.download_weights()
        with open(model_dir / self.weights["classes_file"]) as infile:
            class_names = [line.strip() for line in infile.readlines()]

        self.detect_ids = config["detect"]  # change "detect_ids" to "detect"
        self.detector = Detector(
            model_dir,
            class_names,
            self.detect_ids,
            self.config["model_type"],
            self.weights["model_file"],
            self.config["max_output_size_per_class"],
            self.config["max_total_size"],
            self.config["input_size"],
            self.config["iou_threshold"],
            self.config["score_threshold"],
        )

    @property
    def detect_ids(self) -> List[int]:
        """The list of selected object category IDs."""
        return self._detect_ids

    @detect_ids.setter
    def detect_ids(self, ids: List[int]) -> None:
        if not isinstance(ids, list):
            raise TypeError("detect_ids has to be a list")
        self._detect_ids = ids

    def predict(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts face bboxes, labels and scores

        Args:
            frame (np.ndarray): image in numpy array

        Returns:
            bboxes (np.ndarray): numpy array of detected bboxes
            labels (np.ndarray): numpy array of class labels
            scores (np.ndarray): numpy array of confidence scores
        """
        assert isinstance(frame, np.ndarray)

        return self.detector.predict_object_bbox_from_image(frame)
