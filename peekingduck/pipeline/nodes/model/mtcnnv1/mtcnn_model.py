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


class MTCNNModel(ThresholdCheckerMixin, WeightsDownloaderMixin):
    """MTCNN model to detect face bboxes and landmarks."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.check_bounds("min_size", "(0, +inf]")
        self.check_bounds(
            ["network_thresholds", "scale_factor", "score_threshold"], "[0, 1]"
        )

        model_dir = self.download_weights()
        self.detector = Detector(
            model_dir,
            self.config["model_type"],
            self.weights["model_file"],
            self.config["model_nodes"],
            self.config["min_size"],
            self.config["scale_factor"],
            list(map(float, self.config["network_thresholds"])),
            self.config["score_threshold"],
        )

    def predict(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts face bboxes, scores and landmarks

        Args:
            frame (np.ndarray): image in numpy array

        Returns:
            bboxes (np.ndarray): numpy array of detected bboxes
            scores (np.ndarray): numpy array of confidence scores
            landmarks (np.ndarray): numpy array of facial landmarks
        """
        assert isinstance(frame, np.ndarray)

        return self.detector.predict_object_bbox_from_image(frame)
