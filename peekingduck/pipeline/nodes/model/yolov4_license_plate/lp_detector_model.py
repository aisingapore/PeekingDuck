# Copyright 2021 AI Singapore
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

from peekingduck.pipeline.nodes.model.yolov4_license_plate.licenseplate_files.detector import (
    Detector,
)
from peekingduck.weights_utils import checker, downloader


class Yolov4:  # pylint: disable=too-few-public-methods
    """Yolo model with model types: v4 and v4tiny"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        self.logger = logging.getLogger(__name__)

        # check threshold values
        if not 0 <= config["yolo_iou_threshold"] <= 1:
            raise ValueError("yolo_iou_threshold must be in [0, 1]")

        if not 0 <= config["yolo_score_threshold"] <= 1:
            raise ValueError("yolo_score_threshold must be in [0, 1]")

        # check for yolo(license plate) weights, if none then download
        if not checker.has_weights(config["root"], config["weights_dir"]):
            self.logger.info("---no LP weights detected. proceeding to download...---")
            downloader.download_weights(config["root"], config["blob_file"])
            self.logger.info("---LP weights download complete.---")

        self.detector = Detector(config)

    def predict(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts the bboxes from image frame

        Returns:
                object_bboxes (Numpy Array): list of bboxes detected
                object_labels (Numpy Array): list of string labels of the
                    object detected for the corresponding bbox
                object_scores (Numpy Array): list of confidence scores of the
                    object detected for the corresponding bbox
        """
        assert isinstance(frame, np.ndarray)

        return self.detector.predict_object_bbox_from_image(frame)
