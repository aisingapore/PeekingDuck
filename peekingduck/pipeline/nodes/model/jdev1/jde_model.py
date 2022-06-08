# Modifications copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Original copyright (c) 2019 ZhongdaoWang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""JDE model for human detection and tracking."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from peekingduck.pipeline.nodes.base import (
    ThresholdCheckerMixin,
    WeightsDownloaderMixin,
)
from peekingduck.pipeline.nodes.model.jdev1.jde_files.tracker import Tracker


class JDEModel(ThresholdCheckerMixin, WeightsDownloaderMixin):
    """JDE Model with model types: 576x320, 865x480, and 1088x608.

    Args:
        config (Dict[str, Any]): Model configuration options.
        frame_rate (float): The frame rate of the current video sequence,
            used for computing the size of track buffer.

    Raises:
        ValueError: `iou_threshold` is beyond [0, 1].
        ValueError: `nms_threshold` is beyond [0, 1].
        ValueError: `score_threshold` is beyond [0, 1].
    """

    def __init__(self, config: Dict[str, Any], frame_rate: float) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.check_bounds(
            ["iou_threshold", "nms_threshold", "score_threshold"], "[0, 1]"
        )

        model_dir = self.download_weights()
        self.tracker = Tracker(
            model_dir,
            frame_rate,
            self.config["model_type"],
            self.weights["model_file"],
            self.weights["config_file"],
            self.config["min_box_area"],
            self.config["track_buffer"],
            self.config["iou_threshold"],
            self.config["nms_threshold"],
            self.config["score_threshold"],
        )

    def predict(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[float], List[int]]:
        """Track objects from image.

        Args:
            image (np.ndarray): Image in numpy array.

        Returns:
            (Tuple[List[np.ndarray], List[float], List[int]]): A tuple of
            - Numpy array of detected bounding boxes.
            - List of detection confidence scores.
            - List of track IDs.

        Raises:
            TypeError: The provided `image` is not a numpy array.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a np.ndarray")
        bboxes, track_ids, bbox_scores = self.tracker.track_objects_from_image(image)
        return bboxes, bbox_scores, track_ids
