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

"""Hugging Face Hub model adaptors."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageSegmentation,
    AutoModelForInstanceSegmentation,
    AutoModelForObjectDetection,
)


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


class InstanceSegmenter(HuggingFaceAdaptor):
    """Hugging Face instance segmentation model."""

    def __init__(self, model_type: str, cache_dir: Path, mask_threshold: float) -> None:
        super().__init__(model_type, cache_dir)
        self.mask_threshold = mask_threshold
        self.model = AutoModelForInstanceSegmentation.from_pretrained(
            model_type, cache_dir=cache_dir
        )

    def process_outputs(
        self,
        outputs: Any,
        target_size: Tuple[int, int],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Processes model output into instance segmentation results with
        additional bounding boxes for each detection.

        Args:
            outputs (Any): Model output.
            target_size (Tuple[int, int]): The target size to which the
                ``masks_queries_logits`` will be reshaped. Typically set to the
                original input image size.
            mask_threshold (float): The confidence threshold for binarizing the
                masks' pixel values; determines whether an object is detected
                at a particular pixel.
            device (torch.device): Intermediate tensors should be created on
                this device.

        Returns:
            (Dict[str, torch.Tensor]): A dictionary containing the keys:
            - boxes: Bounding boxes for each detection, with the
              (x1, y1, x2, y2) format, where (x1, y1) is the top-left corner
              and (x2, y2) is the bottom-right corner.
            - labels: Numerical class IDs for each detection.
            - masks: Binary instance masks for each detection.
            - scores: Detections confidence score.
        """
        masks = outputs.masks_queries_logits
        masks = F.interpolate(
            masks, size=target_size, mode="bilinear", align_corners=False
        )
        # Transpose to match the shape of other models
        masks = (masks.sigmoid() > self.mask_threshold).cpu().byte().transpose(0, 1)
        scores, labels = F.softmax(outputs.class_queries_logits, dim=-1).max(-1)
        boxes = self._masks_to_bboxes(masks)
        return {
            "boxes": boxes,
            "labels": labels.squeeze(0),
            "masks": masks,
            "scores": scores.squeeze(0),
        }

    def to(self, device: torch.device) -> None:  # pylint: disable=invalid-name
        """Moves model and various attributes to the specified ``device``.

        Args:
            device (torch.device): The target device.

        Raises:
            AttributeError: If ``model`` does not have a ``model`` attribute.
        """
        if hasattr(self.model, "model"):
            self.model.model.to(device)
            self.model.class_predictor.to(device)
            self.model.mask_embedder.to(device)
            self.model.matcher.to(device)
        else:
            raise AttributeError

    @staticmethod
    def _masks_to_bboxes(masks: torch.Tensor) -> torch.Tensor:
        """Creates tight bounding boxes each encompassing an instance mask.

        Inspired by:
            https://github.com/facebookresearch/detectron2/blob/main/detectron2/structures/masks.py#L224

        Args:
            masks (torch.Tensor): Instance masks.

        Returns:
            (torch.Tensor): Tight bounding boxes each encompassing an instance
            mask.
        """
        boxes = torch.zeros(masks.shape[0], 4, dtype=torch.float32)
        x_any = torch.any(masks, dim=1)
        y_any = torch.any(masks, dim=2)
        for idx in range(masks.shape[0]):
            x_coords = torch.where(x_any[idx, :])[0]
            y_coords = torch.where(y_any[idx, :])[0]
            if len(x_coords) > 0 and len(y_coords) > 0:
                boxes[idx, :] = torch.as_tensor(
                    [x_coords[0], y_coords[0], x_coords[-1] + 1, y_coords[-1] + 1],
                    dtype=torch.float32,
                )
        return boxes


class PanopticSegmenter(HuggingFaceAdaptor):
    """Hugging Face panoptic segmentation model."""

    def __init__(self, model_type: str, cache_dir: Path, mask_threshold: float) -> None:
        super().__init__(model_type, cache_dir)
        self.mask_threshold = mask_threshold
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_type, cache_dir=cache_dir
        )

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
            - masks: Binary instance masks for each detection.
            - scores: Detections confidence score.
        """
        target_sizes = torch.tensor([target_size], device=device)
        results = self.extractor.post_process(
            outputs=outputs, target_sizes=target_sizes
        )
        result = self.extractor.post_process_instance(
            results, outputs, target_sizes, target_sizes, self.mask_threshold
        )[0]
        return result

    def to(self, device: torch.device) -> None:  # pylint: disable=invalid-name
        """Moves model and various attributes to the specified ``device``.

        Args:
            device (torch.device): The target device.

        Raises:
            AttributeError: If ``model`` does not have a ``detr`` attribute.
        """
        if hasattr(self.model, "detr"):
            self.model.detr.to(device)
            self.model.mask_head.to(device)
            self.model.bbox_attention.to(device)
        else:
            raise AttributeError


class ObjectDetector(HuggingFaceAdaptor):
    """Hugging Face object detection model."""

    def __init__(self, model_type: str, cache_dir: Path) -> None:
        super().__init__(model_type, cache_dir)
        self.model = AutoModelForObjectDetection.from_pretrained(
            model_type, cache_dir=cache_dir
        )

    def process_outputs(
        self, outputs: Any, target_size: Tuple[int, int], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Processes model output into object detection results.

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
            - scores: Detections confidence score.
        """
        target_sizes = torch.tensor([target_size], device=device)
        result = self.extractor.post_process(
            outputs=outputs, target_sizes=target_sizes
        )[0]
        return result

    def to(self, device: torch.device) -> None:  # pylint: disable=invalid-name
        """Moves model and various attributes to the specified ``device``.

        Args:
            device (torch.device): The target device.

        Raises:
            AttributeError: If ``model` does not have a ``model`` or ``vit``
            attribute.
        """
        if hasattr(self.model, "model"):
            self.model.model.to(device)
        elif hasattr(self.model, "vit"):
            self.model.vit.to(device)
        else:
            raise AttributeError

        self.model.class_labels_classifier.to(device)
        self.model.bbox_predictor.to(device)
