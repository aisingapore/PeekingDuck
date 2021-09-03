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
from typing import List, Dict, Any, Tuple
import numpy as np
from peekingduck.weights_utils import checker, downloader
from .licenseplate_files.detector import Detector


class Yolov4:
    """Yolo model with model types: v4 and v4tiny"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        self.logger = logging.getLogger(__name__)

        # check threshold values
        if not 0 <= config["confThreshold"] <= 1:
            raise ValueError("confThreshold must be in [0, 1]")

        if not 0 <= config["nmsThreshold"] <= 1:
            raise ValueError("nmsThreshold must be in [0, 1]")

        # check for yolo weights, if none then download into weights folder
        if not checker.has_weights(config["root"], config["weights_dir"]):
            self.logger.info("---no LP weights detected. proceeding to download...---")
            downloader.download_weights(config["root"], config["blob_file"])
            self.logger.info("---LP weights download complete.---")

        self.detector = Detector(config)

    def predict(self, frame: np.array) -> Tuple[List[np.array], List[str], List[float]]:
        """
        predict the bbox from frame

        returns:
        object_bboxes(List[Numpy Array]): list of bboxes detected
        object_labels(List[str]): list of string labels of the
            object detected for the corresponding bbox
        object_scores(List(float)): list of confidence scores of the
            object detected for the corresponding bbox
        """
        assert isinstance(frame, np.ndarray)

        return self.detector.predict_object_bbox_from_image(frame)
