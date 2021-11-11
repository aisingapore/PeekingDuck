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

"""MoveNet model with model types: singlepose lightning/thunder, multipose lightning"""

import logging
from typing import Dict, Any, Tuple
import numpy as np

from peekingduck.weights_utils import checker, downloader
from peekingduck.pipeline.nodes.model.movenetv1.movenet_files.predictor import Predictor


class MoveNetModel:  # pylint: disable=too-few-public-methods
    """MoveNet model with model types: lightning, thunder"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

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
                """model_type must be one of 
                ["singlepose_lightning","singlepose_thunder","multipose_lightning"]"""
            )

        # check for movenet weights, if none then download into weights folder
        if not checker.has_weights(config["root"], config["weights_dir"]):
            print("---no movenet weights detected. proceeding to download...---")
            downloader.download_weights(config["root"], config["blob_file"])
            print("---movenet weights download complete.---")

        self.predictor = Predictor(config)

    def predict(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Predict poses from input frame

        Args:
            frame (np.array): image in numpy array

        Returns:
            bboxes, keypoints, keypoint_scores, keypoint_masks, keypoint_conns
            (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]): \
            tuple containing list of bboxes and pose related info i.e coordinates,
            scores, connections
        """
        assert isinstance(frame, np.ndarray)

        return self.predictor.predict(frame)
