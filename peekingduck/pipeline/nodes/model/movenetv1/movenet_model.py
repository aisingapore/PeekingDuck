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

"""MoveNet model with model types: singlepose lightning/thunder, multipose lightning"""

import logging
from typing import Any, Dict, Tuple

import numpy as np

from peekingduck.pipeline.nodes.base import (
    ThresholdCheckerMixin,
    WeightsDownloaderMixin,
)
from peekingduck.pipeline.nodes.model.movenetv1.movenet_files.predictor import Predictor


class MoveNetModel(ThresholdCheckerMixin, WeightsDownloaderMixin):
    """MoveNet model with model types: lightning, thunder."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.check_valid_choice(
            "model_type",
            {"singlepose_lightning", "singlepose_thunder", "multipose_lightning"},
        )
        self.check_bounds(
            ["bbox_score_threshold", "keypoint_score_threshold"], "[0, 1]"
        )

        model_dir = self.download_weights()
        self.predictor = Predictor(
            model_dir,
            self.config["model_type"],
            self.weights["model_file"],
            self.config["resolution"],
            self.config["bbox_score_threshold"],
            self.config["keypoint_score_threshold"],
        )

    def predict(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict poses from input frame

        Args:
            frame (np.array): image in numpy array

        Returns:
            Tuple of outputs of the model
            (bboxes, keypoints, keypoint_scores, keypoint_masks, keypoint_conns)

            bboxes (np.ndarray): Nx4 array of bboxes, N is number of detections
            keypoints (np.ndarray): Nx17x2 array of keypoint coordinates
            keypoints_scores (np.ndarray): Nx17 array of keypoint scores
            keypoints_conns (np.ndarray): NxD'x2 keypoint connections, where
                D' is the varying pair of valid keypoint connections per detection
        """
        return self.predictor.predict(frame)
