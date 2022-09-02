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

import logging
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torchvision
from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForObjectDetection

from peekingduck.pipeline.nodes.base import (
    ThresholdCheckerMixin,
    WeightsDownloaderMixin,
)
from peekingduck.pipeline.nodes.model.huggingfacev1.api_utils import get_valid_models
from peekingduck.pipeline.utils.bbox.transforms import xyxy2xyxyn


class ObjectDetectionModel(ThresholdCheckerMixin, WeightsDownloaderMixin):
    """Validates configuration, loads model from Hugging Face Hub, and performs
    inference.

    Configuration options are validated to ensure they have valid types and
    values. Model weights files are downloaded if not found in the location
    indicated by the `weights_dir` configuration option.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.check_bounds(["iou_threshold", "score_threshold"], "[0, 1]")
        self.check_valid_choice("task", {"object_detection"})
        self.check_valid_choice("model_type", get_valid_models(self.config["task"]))  # type: ignore

        self.iou_threshold = self.config["iou_threshold"]
        self.score_threshold = self.config["score_threshold"]
        self.agnostic_nms = self.config["agnostic_nms"]

        self.model_type_parts = self.config["model_type"].split("/")
        model_dir = self.prepare_cache_dir()

        self.detector_config = AutoConfig.from_pretrained(
            self.config["model_type"], cache_dir=model_dir
        )
        print(self.detector_config.label2id)

        self.extractor = AutoFeatureExtractor.from_pretrained(
            self.config["model_type"], cache_dir=model_dir
        )
        self.detector = AutoModelForObjectDetection.from_pretrained(
            self.config["model_type"], cache_dir=model_dir
        )

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts bboxes from image.

        Args:
            image (np.ndarray): Input image frame.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains:
            - An array of detection bboxes
            - An array of human-friendly detection class names
            - An array of detection scores

        Raises:
            TypeError: The provided `image` is not a numpy array.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a np.ndarray")

        image_size = image.shape[0], image.shape[1]
        inputs = self.extractor(images=image, return_tensors="pt")
        outputs = self.detector(**inputs)

        target_sizes = torch.tensor([image.shape[:2]])
        result = self.extractor.post_process(
            outputs=outputs, target_sizes=target_sizes
        )[0]

        return self._postprocess(result, image_size)

    def _postprocess(
        self, result: Dict[str, torch.Tensor], image_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        detections = torch.cat(
            (
                result["boxes"],
                result["scores"].unsqueeze(-1),
                result["labels"].unsqueeze(-1),
            ),
            1,
        )
        # Filter by score_threshold
        detections = detections[detections[:, 4] >= self.score_threshold]
        # Early return if all are below score_threshold
        if detections.size(0) == 0:
            return np.empty((0, 4)), np.empty(0), np.empty(0)

        # Class agnostic NMS
        if self.agnostic_nms:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                self.iou_threshold,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                self.iou_threshold,
            )
        detections = detections[nms_out_index]

        detections_np = detections.cpu().detach().numpy()
        bboxes = xyxy2xyxyn(detections_np[:, :4], *image_shape)
        scores = detections_np[:, 4]
        classes = np.array(
            [self.detector.config.id2label[int(i)] for i in detections_np[:, 5]]
        )

        return bboxes, classes, scores
