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
Face detection model with model types: yolov4 and yolov4tiny
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from peekingduck.weights_utils import checker, downloader, finder
from peekingduck.pipeline.nodes.model.yolov4_face.yolo_face_files.detector import (
    Detector,
)


class Yolov4:  # pylint: disable=too-few-public-methods
    """Yolo model with model types: v4 and v4tiny"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        self.logger = logging.getLogger(__name__)

        # check threshold values
        if not 0 <= config["yolo_score_threshold"] <= 1:
            raise ValueError("yolo_score_threshold must be in [0, 1]")

        if not 0 <= config["yolo_iou_threshold"] <= 1:
            raise ValueError("yolo_iou_threshold must be in [0, 1]")

        weights_dir, model_dir = finder.find_paths(
            config["root"], config["weights"], config["weights_parent_dir"]
        )

        # check for yolo weights, if none then download into weights folder
        if not checker.has_weights(weights_dir, model_dir):
            self.logger.info("---no weights detected. proceeding to download...---")
            downloader.download_weights(weights_dir, config["weights"]["blob_file"])
            self.logger.info(f"---weights downloaded to {weights_dir}.---")

        self.detect_ids = config["detect_ids"]
        self.detector = Detector(config, model_dir)

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

    def get_detect_ids(self) -> List[int]:
        """Getter for selected ids for detection. This function is used in unit
        testing.

        Returns:
            List[int]: list of selected detection ids
        """
        return self.detect_ids
