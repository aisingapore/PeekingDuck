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

"""Hugging Face Hub models."""

from typing import Any, Dict, Tuple

import numpy as np
import torch
from transformers import AutoConfig

from peekingduck.nodes.model.huggingface_hubv1 import adaptors
from peekingduck.nodes.model.huggingface_hubv1.models import base
from peekingduck.utils.bbox.transforms import xyxy2xyxyn


class InstanceSegmentationModel(base.HuggingFaceModel):
    """Validates configuration, loads image segmentation models from Hugging
    Face Hub, and performs inference.

    Configuration options are validated to ensure they have valid types and
    values. Model weights files are downloaded if not found in the location
    indicated by the `weights_dir` configuration option.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.check_bounds("mask_threshold", "[0, 1]")

        self.mask_threshold = self.config["mask_threshold"]

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Predicts bboxes from image.

        Args:
            image (np.ndarray): Input image frame.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains:
            - An array of detection bboxes
            - An array of human-friendly detection class names
            - An array of detection scores
            - An array of detection masks

        Raises:
            TypeError: The provided `image` is not a numpy array.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a np.ndarray")
        image_size = image.shape[0], image.shape[1]
        result = self.adaptor.predict(image, self.device)

        return self._postprocess(result, image_size)

    def _create_huggingface_model(self, model_type: str) -> None:
        """Creates the specified Hugging Face model from pretrained weights.
        In addition, also sets up cache directory paths, maps `detect` list to
        numeric detect IDs, and attempts to setup inference on GPU.
        """
        model_dir = self.prepare_cache_dir(model_type.split("/"))

        detector_config = AutoConfig.from_pretrained(model_type, cache_dir=model_dir)
        detect_ids = self._map_detect_ids(detector_config, self.config["detect"])

        try:
            self.adaptor = adaptors.PanopticSegmenter(
                model_type, model_dir, self.mask_threshold
            )
        except ValueError:
            self.adaptor = adaptors.InstanceSegmenter(
                model_type, model_dir, self.mask_threshold
            )
        self.device = self._init_device()
        self.detect_ids = detect_ids

    def _log_model_configs(self) -> None:
        """Prints the loaded model's settings."""
        self.logger.info(
            "Hugging Face model loaded with the following configs:\n\t"
            f"Model type: {self.config['model_type']}\n\t"
            f"IDs being detected: {self.detect_ids}\n\t"
            f"Mask threshold: {self.mask_threshold}\n\t"
            f"Score threshold: {self.score_threshold}"
        )

    def _postprocess(
        self, result: Dict[str, torch.Tensor], image_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, ...]:
        """Post processes detection result. Filters detection results by
        confidence score and discards detections which do not contain objects
        of interest based on ``self._detect_ids``.

        Args:
            result (Dict[str, torch.Tensor]): A dictionary containing the model
                output, with the keys: "boxes", "labels", "masks, and "scores".
            image_shape (Tuple[int, int]): The height and width of the input
                image.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains:
            - An array of detection bboxes
            - An array of human-friendly detection class names
            - An array of detection scores
            - An array of binary instance masks
        """
        if result["masks"].size(0) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty(0),
                np.empty(0, dtype=np.float32),
                np.empty((0, 0, 0), dtype=np.uint8),
            )
        detect_filter = torch.isin(result["labels"], self._detect_ids)
        score_filter = result["scores"] > self.score_threshold
        for key in result:
            result[key] = (
                result[key][detect_filter & score_filter].cpu().detach().numpy()
            )
        bboxes = xyxy2xyxyn(result["boxes"], *image_shape)
        classes = np.array([self.adaptor.id2label[int(i)] for i in result["labels"]])
        return bboxes, classes, result["scores"], result["masks"].squeeze(1)
