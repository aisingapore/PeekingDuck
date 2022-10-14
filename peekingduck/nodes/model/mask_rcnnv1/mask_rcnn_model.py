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
Mask R-CNN model with backbone type: r50-fpn and r101-fpn
"""

import logging
from typing import Any, Dict, List, Tuple
import json
import numpy as np

from peekingduck.pipeline.nodes.base import (
    ThresholdCheckerMixin,
    WeightsDownloaderMixin,
)
from peekingduck.pipeline.nodes.model.mask_rcnnv1.mask_rcnn_files.detector import (
    Detector,
)


class MaskRCNNModel(ThresholdCheckerMixin, WeightsDownloaderMixin):
    """Mask R-CNN model with ResNet 50 / 101 FPN backbone"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.check_bounds(
            ["iou_threshold", "score_threshold", "mask_threshold"], "[0, 1]"
        )
        self.check_bounds(["min_size", "max_size", "max_num_detections"], "[1 , +inf)")

        model_dir = self.download_weights()
        classes_path = model_dir / self.weights["classes_file"]
        class_names = {
            val["id"] - 1: val["name"]
            for val in json.loads(classes_path.read_text()).values()
        }

        self.detect_ids = self.config["detect"]
        self.detector = Detector(
            model_dir,
            class_names,
            self.detect_ids,
            self.config["model_type"],
            self.config["num_classes"],
            self.weights["model_file"],
            self.config["min_size"],
            self.config["max_size"],
            self.config["iou_threshold"],
            self.config["max_num_detections"],
            self.config["score_threshold"],
            self.config["mask_threshold"],
        )

    @property
    def detect_ids(self) -> List[int]:
        """The list of selected object category IDs."""
        return self._detect_ids

    @detect_ids.setter
    def detect_ids(self, ids: List[int]) -> None:
        if not isinstance(ids, list):
            raise TypeError("detect_ids has to be a list")
        if not ids:
            self.logger.info("Detecting all Mask R-CNN classes")
        self._detect_ids = ids

    def predict(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predicts bboxes and masks from image.

        Args:
            image (np.ndarray): Input image frame.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains:
            - An array of detection bboxes
            - An array of human-friendly detection class names
            - An array of detection scores
            - An array of binarized masks

        Raises:
            TypeError: The provided `image` is not a numpy array.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a np.ndarray")
        return self.detector.predict_instance_mask_from_image(image)
