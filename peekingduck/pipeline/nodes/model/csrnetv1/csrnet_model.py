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
CSRNet model with model types: sparse and dense
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np

from peekingduck.pipeline.nodes.base import (
    ThresholdCheckerMixin,
    WeightsDownloaderMixin,
)
from peekingduck.pipeline.nodes.model.csrnetv1.csrnet_files.predictor import Predictor


class CSRNetModel(ThresholdCheckerMixin, WeightsDownloaderMixin):
    """CSRNet model with model types: sparse and dense"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.check_bounds("width", "(0, +inf]")

        model_dir = self.download_weights()
        self.predictor = Predictor(
            model_dir,
            self.config["model_type"],
            self.weights["model_file"],
            self.config["width"],
        )

    def predict(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """Predicts density map and crowd count from frame.

        Args:
            frame (np.ndarray): input frame.

        Returns:
            density_map (np.ndarray): predicted density map.
            crowd_count (int): predicted count of people.
        """
        assert isinstance(frame, np.ndarray)

        # return density_map, crowd_count
        return self.predictor.predict_count_from_image(frame)
