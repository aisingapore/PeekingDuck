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

from peekingduck.pipeline.nodes.model.movenetv1.movenet_files.predictor import Predictor
from peekingduck.weights_utils import checker, downloader, finder


class MoveNetModel:  # pylint: disable=too-few-public-methods
    """MoveNet model with model types: lightning, thunder"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = logging.getLogger(__name__)

        # check threshold values
        if not 0 <= config["bbox_score_threshold"] <= 1:
            raise ValueError("bbox_score_threshold must be in [0, 1]")
        if not 0 <= config["keypoint_score_threshold"] <= 1:
            raise ValueError("keypoint_score_threshold must be in [0, 1]")
        if config["model_type"] not in [
            "singlepose_lightning",
            "singlepose_thunder",
            "multipose_lightning",
        ]:
            raise ValueError(
                "model_type must be one of ['singlepose_lightning', "
                "'singlepose_thunder', 'multipose_lightning']"
            )

        weights_dir, model_dir = finder.find_paths(
            config["root"], config["weights"], config["weights_parent_dir"]
        )

        # check for movenet weights, if none then download into weights folder
        if not checker.has_weights(weights_dir, model_dir):
            self.logger.info("---no weights detected. proceeding to download...---")
            downloader.download_weights(weights_dir, config["weights"]["blob_file"])
            self.logger.info(f"---weights downloaded to {weights_dir}.---")

        self.predictor = Predictor(config, model_dir)

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
