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
Yolo model with model types: v4 and v4tiny
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from peekingduck.pipeline.nodes.base import (
    ThresholdCheckerMixin,
    WeightsDownloaderMixin,
)
from peekingduck.pipeline.nodes.model.yolov4.yolo_files.detector import Detector


class YoloModel(ThresholdCheckerMixin, WeightsDownloaderMixin):
    """Yolo model with model types: v4 and v4tiny"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.ensure_within_bounds(["iou_threshold", "score_threshold"], 0, 1)

        model_dir = self.download_weights()
        with open(model_dir / config["weights"]["classes_file"]) as infile:
            self.class_names = [c.strip() for c in infile.readlines()]

        self.detector = Detector(config, model_dir, self.class_names)
        self.detect_ids = config["detect_ids"]

    @property
    def detect_ids(self) -> List[int]:
        """The list of selected object category IDs."""
        return self._detect_ids

    @detect_ids.setter
    def detect_ids(self, ids: List[int]) -> None:
        if not isinstance(ids, list):
            raise TypeError("detect_ids has to be a list")
        if not ids:
            self.logger.info("Detecting all Yolo classes")
        self._detect_ids = ids

    def predict(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[str], List[float]]:
        """predict the bbox from frame

        Args:
            image (np.ndarray): Input image frame.

        Returns:
            object_bboxes (List[Numpy Array]): list of bboxes detected
            object_labels (List[str]): list of string labels of the
                object detected for the corresponding bbox
            object_scores (List(float)): list of confidence scores of the
                object detected for the corresponding bbox
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a np.ndarray")

        # return object_bboxes, object_labels, object_scores
        return self.detector.predict_object_bbox_from_image(image, self.detect_ids)
