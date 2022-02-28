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
Main class for HRNet Model
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np

from peekingduck.pipeline.nodes.model.hrnetv1.hrnet_files.detector import Detector
from peekingduck.weights_utils import checker, downloader, finder


class HRNetModel:  # pylint: disable=too-few-public-methods
    """HRNet model to detect poses from detected bboxes."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        self.logger = logging.getLogger(__name__)

        # check threshold values
        if not 0 <= config["score_threshold"] <= 1:
            raise ValueError("score_threshold must be in [0, 1]")

        weights_dir, model_dir = finder.find_paths(
            config["root"], config["weights"], config["weights_parent_dir"]
        )

        # check for hrnet weights, if none then download into weights folder
        if not checker.has_weights(weights_dir, model_dir):
            self.logger.info("---no weights detected. proceeding to download...---")
            downloader.download_weights(weights_dir, config["weights"]["blob_file"])
            self.logger.info(f"---weights downloaded to {weights_dir}.---")

        self.detector = Detector(config, model_dir)
        self.threshold_score = config["score_threshold"]

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
