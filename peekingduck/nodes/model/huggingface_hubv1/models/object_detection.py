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

"""Hugging Face Hub object detection model."""

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torchvision
from transformers import AutoConfig

from peekingduck.nodes.model.huggingface_hubv1 import adaptors
from peekingduck.nodes.model.huggingface_hubv1.models import base
from peekingduck.utils.bbox.transforms import xyxy2xyxyn


class ObjectDetectionModel(base.HuggingFaceModel):
    """Validates configuration, loads model from Hugging Face Hub, and performs
    inference.

    Configuration options are validated to ensure they have valid types and
    values. Model weights files are downloaded if not found in the location
    indicated by the `weights_dir` configuration option.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.check_bounds("iou_threshold", "[0, 1]")

        self.iou_threshold = self.config["iou_threshold"]
        self.agnostic_nms = self.config["agnostic_nms"]

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

        self.adaptor = adaptors.ObjectDetector(model_type, model_dir)
        self.device = self._init_device()
        self.detect_ids = detect_ids

    def _log_model_configs(self) -> None:
        """Prints the loaded model's settings."""
        self.logger.info(
            "Hugging Face model loaded with the following configs:\n\t"
            f"Model type: {self.config['model_type']}\n\t"
            f"IDs being detected: {self.detect_ids}\n\t"
            f"IOU threshold: {self.iou_threshold}\n\t"
            f"Score threshold: {self.score_threshold}\n\t"
            f"Class agnostic NMS: {self.agnostic_nms}"
        )

    def _postprocess(
        self, result: Dict[str, torch.Tensor], image_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, ...]:
        """Post-processes model output. Filters detection results by confidence
        score, performs non-maximum suppression, and discards detections which
        do not contain objects of interest based on ``self._detect_ids``.

        Args:
            result (Dict[str, torch.Tensor]): A dictionary containing the model
                output, with the keys: "boxes", "scores", and "labels".
            image_shape (Tuple[int, int]): The height and width of the input
                image.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains:
            - An array of detection bboxes
            - An array of human-friendly detection class names
            - An array of detection scores
        """
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
                detections[:, :4], detections[:, 4], self.iou_threshold
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4],
                detections[:, 5],
                self.iou_threshold,
            )
        detections = detections[nms_out_index]

        if self._detect_ids.size(0) > 0:
            detections = detections[torch.isin(detections[:, 5], self._detect_ids)]
        detections_np = detections.cpu().detach().numpy()
        bboxes = xyxy2xyxyn(detections_np[:, :4], *image_shape)
        scores = detections_np[:, 4]
        classes = np.array([self.adaptor.id2label[int(i)] for i in detections_np[:, 5]])

        return bboxes, classes, scores
