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
Main class for MTCNN Model
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np

from peekingduck.pipeline.nodes.base import (
    ThresholdCheckerMixin,
    WeightsDownloaderMixin,
)
from peekingduck.pipeline.nodes.model.mtcnnv1.mtcnn_files.detector import Detector


class MTCNNModel(
    ThresholdCheckerMixin, WeightsDownloaderMixin
):  # pylint: disable=too-few-public-methods
    """MTCNN model to detect face bboxes and landmarks"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        # check threshold values
        for threshold in config["network_thresholds"]:
            if not 0 <= threshold <= 1:
                raise ValueError("network_thresholds must be in [0, 1]")

        self.ensure_above_value("min_size", 0)
        self.ensure_within_bounds(["scale_factor", "score_threshold"], 0, 1)

        model_dir = self.download_weights()
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

        # return bboxes, scores, landmarks and class labels
        return self.detector.predict_bbox_landmarks(frame)
