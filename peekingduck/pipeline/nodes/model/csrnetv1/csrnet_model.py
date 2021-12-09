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
CSRNet model with model types: sparse and dense
"""

import logging
from typing import Dict, Any, Tuple

import numpy as np

from peekingduck.pipeline.nodes.model.csrnetv1.csrnet_files.predictor import Predictor
from peekingduck.weights_utils import checker, downloader, finder


class CsrnetModel:  # pylint: disable=too-few-public-methods
    """CSRNet model with model types: sparse and dense"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)

        weights_dir, model_dir = finder.find_paths(
            config["root"], config["weights"], config["weights_parent_dir"]
        )

        # check for csrnet weights, if none then download into weights folder
        if not checker.has_weights(weights_dir, model_dir):
            self.logger.info("---no weights detected. proceeding to download...---")
            downloader.download_weights(weights_dir, config["weights"]["blob_file"])
            self.logger.info(f"---weights downloaded to {weights_dir}.---")

        self.predictor = Predictor(config, model_dir)

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
