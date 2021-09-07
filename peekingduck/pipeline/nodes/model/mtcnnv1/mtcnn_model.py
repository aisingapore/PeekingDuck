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

import logging
from typing import Dict, Any, Tuple

import numpy as np

from peekingduck.weights_utils import checker, downloader
from .mtcnn_files.detector import Detector


class MtcnnModel:
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        self.logger = logging.getLogger(__name__)

        # check factor values
        for factor in config['mtcnn_factors']:
            if not factor < 1:
               raise ValueError("mtcnn_factors must be less than 1")             

        # check threshold values
        if not 0 <= config['mtcnn_threshold'] <= 1:
            raise ValueError("mtcnn_threshold must be between 0 and 1")

        # check for mtcnn weights, if none then download into weights folder
        if not checker.has_weights(config['root'],
                                   config['weights_dir']):
            self.logger.info('---no mtcnn weights detected. proceeding to download...---')
            downloader.download_weights(config['root'],
                                        config['blob_file'])
            self.logger.info('---mtcnn weights download complete.---')

        self.detector = Detector(config)    

    def predict(self, frame: np.array) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert isinstance(frame, np.ndarray)

        # return bboxes, scores and landmarks
        return self.detector.predict_bbox_landmarks(frame)