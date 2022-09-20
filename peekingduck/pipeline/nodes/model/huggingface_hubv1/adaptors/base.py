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

"""Base Hugging Face Hub model adaptors."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from transformers import AutoFeatureExtractor


class HuggingFaceAdaptor(ABC):
    """Adaptor for Hugging Face Hub models."""

    # Created from AutoFeatureExtractor
    extractor: Any
    # Created from either AutoModelForImageSegmentation,
    # AutoModelForInstanceSegmentation, or AutoModelForObjectDetection
    model: Any

    def __init__(self, model_type: str, cache_dir: Path) -> None:
        self.extractor = AutoFeatureExtractor.from_pretrained(
            model_type, cache_dir=cache_dir
        )

    @property
    def id2label(self) -> Dict[int, str]:
        """Returns the detect ID to class label mapping."""
        return self.model.config.id2label

    @abstractmethod
    def process_outputs(
        self, outputs: Any, target_size: Tuple[int, int], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Processes model output into instance segmentation results with
        additional bounding boxes for each detection.

        Args:
            outputs (Any): Model output.
            target_size (Tuple[int, int]): The target size to which the
                ``masks_queries_logits`` will be reshaped. Typically set to the
                original input image size.
            device (torch.device): Intermediate tensors should be created on
                this device.

        Returns:
            (Dict[str, torch.Tensor]): A dictionary containing the keys:
            - boxes: Bounding boxes for each detection, with the
              (x1, y1, x2, y2) format, where (x1, y1) is the top-left corner
              and (x2, y2) is the bottom-right corner.
            - labels: Numerical class IDs for each detection.
            - masks: Binary instance masks for each detection. Only for
              segmentation models
            - scores: Detections confidence score.
        """

    @abstractmethod
    def to(self, device: torch.device) -> None:  # pylint: disable=invalid-name
        """Moves model and various attributes to the specified ``device``.

        Args:
            device (torch.device): The target device.

        Raises:
            AttributeError: If ``model` does not have certain attributes, to be
            specified by concrete class.
        """

    def predict(
        self, image: np.ndarray, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Predicts detections from image.

        Args:
            image (np.ndarray): Input image frame.
            device (torch.device): Intermediate tensors should be created on
                this device.

        Returns:
            (Dict[str, torch.Tensor]): A dictionary containing the keys:
            - boxes: Bounding boxes for each detection, with the
              (x1, y1, x2, y2) format, where (x1, y1) is the top-left corner
              and (x2, y2) is the bottom-right corner.
            - labels: Numerical class IDs for each detection.
            - masks: Binary instance masks for each detection, if image
              segmentation model
            - scores: Detections confidence score.

        Raises:
            TypeError: The provided `image` is not a numpy array.
        """
        image_size = image.shape[0], image.shape[1]
        inputs = self.extractor(images=image, return_tensors="pt").to(device)
        outputs = self.model(**inputs)

        return self.process_outputs(outputs, image_size, device)
