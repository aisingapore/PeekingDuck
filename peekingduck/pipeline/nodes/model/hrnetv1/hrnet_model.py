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

"""Main class for HRNet Model."""

import logging
from typing import Any, Dict, Tuple

import numpy as np

from peekingduck.pipeline.nodes.base import (
    ThresholdCheckerMixin,
    WeightsDownloaderMixin,
)
from peekingduck.pipeline.nodes.model.hrnetv1.hrnet_files.detector import Detector


class HRNetModel(ThresholdCheckerMixin, WeightsDownloaderMixin):
    """HRNet model to detect poses from detected bboxes."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.check_bounds("score_threshold", "[0, 1]")

        model_dir = self.download_weights()
        self.detector = Detector(
            model_dir,
            self.config["model_type"],
            self.weights["model_file"],
            self.config["model_nodes"],
            self.config["resolution"],
            self.config["score_threshold"],
        )

    def predict(
        self, frame: np.ndarray, bboxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts poses from input frame and bboxes.

        Args:
            frame (np.ndarray): Image in numpy array.
            bboxes (np.ndarray): Detected bboxes in image.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Tuple containing list
            of pose related info, i.e., coordinates, scores, and connections.
        """
        assert isinstance(frame, np.ndarray)
        assert isinstance(bboxes, np.ndarray)
        detected_bboxes = bboxes.copy()
        if bboxes.size != 0:
            return self.detector.predict(frame, detected_bboxes)
        keypoints = np.array([])
        keypoint_scores = np.array([])
        keypoint_conns = np.array([])

        return keypoints, keypoint_scores, keypoint_conns
