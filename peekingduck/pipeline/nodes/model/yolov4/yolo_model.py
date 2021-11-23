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
Yolo model with model types: v3 and v3tiny
"""

import logging
from typing import List, Dict, Any, Tuple

import numpy as np

from peekingduck.pipeline.nodes.model.yolov4.yolo_files.detector import Detector
from peekingduck.weights_utils import checker, downloader, finder


class YoloModel:
    """Yolo model with model types: v4 and v4tiny"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # check threshold values
        if not 0 <= config["yolo_score_threshold"] <= 1:
            raise ValueError("yolo_score_threshold must be in [0, 1]")

        weights_dir, model_dir = finder.find_paths(
            config["root"], config["weights"], config["weights_parent_dir"]
        )

        # check for yolo weights, if none then download into weights folder
        if not checker.has_weights(weights_dir, model_dir):
            self.logger.info("---no weights detected. proceeding to download...---")
            downloader.download_weights(weights_dir, config["weights"]["blob_file"])
            self.logger.info(f"---weights downloaded to {weights_dir}.---")

        classes_path = model_dir / config["weights"]["classes_file"]

        with open(classes_path) as infile:
            self.class_names = [c.strip() for c in infile.readlines()]
        self.detect_ids = config["detect_ids"]

        self.detector = Detector(config, model_dir)

    def predict(
        self, frame: np.ndarray
    ) -> Tuple[List[np.ndarray], List[str], List[float]]:
        """predict the bbox from frame

        Returns:
            object_bboxes (List[Numpy Array]): list of bboxes detected
            object_labels (List[str]): list of string labels of the
                object detected for the corresponding bbox
            object_scores (List(float)): list of confidence scores of the
                object detected for the corresponding bbox
        """
        assert isinstance(frame, np.ndarray)

        # return bboxes, object_bboxes, object_labels, object_scores
        return self.detector.predict_object_bbox_from_image(
            self.class_names, frame, self.detect_ids
        )

    def get_detect_ids(self) -> List[int]:
        """getter for selected ids for detection

        Returns:
            List[int]: list of selected detection ids
        """
        return self.detect_ids
