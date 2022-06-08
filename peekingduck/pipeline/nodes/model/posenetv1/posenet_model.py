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

"""PoseNet model with model types: mobilenet50, mobilenet75, mobilenet100 and
resnet.
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np

from peekingduck.pipeline.nodes.base import (
    ThresholdCheckerMixin,
    WeightsDownloaderMixin,
)
from peekingduck.pipeline.nodes.model.posenetv1.posenet_files.predictor import Predictor


class PoseNetModel(ThresholdCheckerMixin, WeightsDownloaderMixin):
    """PoseNet model with model types: mobilenet50, mobilenet75, mobilenet100
    and resnet.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.check_valid_choice("model_type", {50, 75, 100, "resnet"})
        self.check_bounds("score_threshold", "[0, 1]")

        model_dir = self.download_weights()
        self.predictor = Predictor(
            model_dir,
            self.config["model_type"],
            self.weights["model_file"],
            self.config["model_nodes"],
            self.config["resolution"],
            self.config["max_pose_detection"],
            self.config["score_threshold"],
        )

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
