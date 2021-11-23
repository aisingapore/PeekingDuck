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
Main class for MTCNN Model
"""

import logging
from typing import Dict, Any, Tuple

import numpy as np

from peekingduck.weights_utils import checker, downloader, finder
from peekingduck.pipeline.nodes.model.mtcnnv1.mtcnn_files.detector import Detector


class MtcnnModel:  # pylint: disable=too-few-public-methods
    """MTCNN model to detect face bboxes and landmarks"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        self.logger = logging.getLogger(__name__)

        # check factor value
        if not 0 <= config["mtcnn_factor"] <= 1:
            raise ValueError("mtcnn_factor must be between 0 and 1")

        # check threshold values
        for threshold in config["mtcnn_thresholds"]:
            if not 0 <= threshold <= 1:
                raise ValueError("mtcnn_thresholds must be between 0 and 1")

        # check score value
        if not 0 <= config["mtcnn_score"] <= 1:
            raise ValueError("mtcnn_score must be between 0 and 1")

        weights_dir, model_dir = finder.find_paths(
            config["root"], config["weights"], config["weights_parent_dir"]
        )

        # check for mtcnn weights, if none then download into weights folder
        if not checker.has_weights(weights_dir, model_dir):
            self.logger.info("---no weights detected. proceeding to download...---")
            downloader.download_weights(weights_dir, config["weights"]["blob_file"])
            self.logger.info(f"---weights downloaded to {weights_dir}.---")

        self.detector = Detector(config, model_dir)

    def predict(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predicts face bboxes, scores and landmarks

        Args:
            frame (np.ndarray): image in numpy array

        Returns:
            bboxes (np.ndarray): numpy array of detected bboxes
            scores (np.ndarray): numpy array of confidence scores
            landmarks (np.ndarray): numpy array of facial landmarks
            labels (np.ndarray): numpy array of class labels (i.e. face)
        """
        assert isinstance(frame, np.ndarray)

        # return bboxes, scores, landmarks amd class labels
        return self.detector.predict_bbox_landmarks(frame)
