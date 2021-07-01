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
EfficientDet model with model types: D0-D4
"""

import logging
from typing import Dict, Any, List, Tuple
import numpy as np

from peekingduck.weights_utils import checker, downloader
from peekingduck.pipeline.nodes.model.efficientdet_d04.efficientdet_files.detector import Detector


class EfficientDetModel:
    """EfficientDet model with model types: D0-D4"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        self.logger = logging.getLogger(__name__)
        # check for efficientdet weights, if none then download into weights folder
        if not checker.has_weights(config['root'],
                                   config['weights_dir']):
            self.logger.info('---no efficientdet weights detected. proceeding to download...---')
            downloader.download_weights(config['root'],
                                        config['blob_file'])
            self.logger.info('---efficientdet weights download complete.---')

        self.detect_ids = config['detect_ids']
        self.logger.info('efficientdet model detecting ids: %s', self.detect_ids)

        self.detector = Detector(config)

    def predict(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """predict the bbox from frame

        returns:
        object_bboxes(np.ndarray): list of bboxes detected
        object_labels(np.ndarray): list of index labels of the
            object detected for the corresponding bbox
        object_scores(np.ndarray): list of confidence scores of the
            object detected for the corresponding bbox
        """
        assert isinstance(frame, np.ndarray)

        # return bboxes, object_bboxes, object_labels, object_scores
        return self.detector.predict_bbox_from_image(frame, self.detect_ids)

    def get_detect_ids(self) -> List[int]:
        """getter function for ids to be detected
        """
        return self.detect_ids
