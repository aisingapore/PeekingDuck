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
Main class for HRNet Model
"""


import logging
from typing import Any, Dict, Tuple

import numpy as np
from peekingduck.weights_utils import checker, downloader
from peekingduck.pipeline.nodes.model.hrnetv1.hrnet_files.detector import Detector


class HRNetModel:  # pylint: disable=too-few-public-methods
    """HRNet model to detect poses from detected bboxes
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        self.logger = logging.getLogger(__name__)

        # check for hrnet weights, if none then download into weights folder
        if not checker.has_weights(config['root'], config['weights_dir']):
            print('---no hrnet weights detected. proceeding to download...---')
            downloader.download_weights(config['root'], config['blob_file'])
            print('---hrnet weights download complete.---')

        self.detector = Detector(config)
        self.threshold_score = config['score_threshold']

    def predict(self, frame: np.ndarray, bboxes: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict poses from input frame and bboxes

        Args:
            frame (np.array): image in numpy array
            bboxes (np.array): detected bboxes in image

        Returns:
            keypoints, keypoint_scores, keypoint_conns, keypoint_bboxes
            (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]): \
            tuple containing list of bboxes and pose related info i.e coordinates,
            scores, connections
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
