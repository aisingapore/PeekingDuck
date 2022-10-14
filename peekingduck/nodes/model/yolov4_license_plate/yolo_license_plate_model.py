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
License Plate detection model with model types: yolov4 and yolov4tiny
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np

from peekingduck.pipeline.nodes.base import (
    ThresholdCheckerMixin,
    WeightsDownloaderMixin,
)
from peekingduck.pipeline.nodes.model.yolov4_license_plate.yolo_license_plate_files.detector import (  # pylint: disable=line-too-long
    Detector,
)


class YOLOLicensePlateModel(ThresholdCheckerMixin, WeightsDownloaderMixin):
    """Yolo model with model types: v4 and v4tiny"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.check_bounds(["iou_threshold", "score_threshold"], "[0, 1]")

        model_dir = self.download_weights()
        with open(model_dir / self.weights["classes_file"]) as infile:
            class_names = [line.strip() for line in infile.readlines()]

        self.detector = Detector(
            model_dir,
            class_names,
            self.config["model_type"],
            self.weights["model_file"],
            self.config["max_output_size_per_class"],
            self.config["max_total_size"],
            self.config["input_size"],
            self.config["iou_threshold"],
            self.config["score_threshold"],
        )

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts the bboxes from image frame

        Args:
            image (np.ndarray): Input image frame.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains:
            - A list of detection bboxes
            - A list of human-friendly detection class names
            - A list of detection scores
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a np.ndarray")

        return self.detector.predict_object_bbox_from_image(image)
