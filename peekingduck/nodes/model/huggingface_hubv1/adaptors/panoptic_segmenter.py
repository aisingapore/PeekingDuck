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

"""Hugging Face Hub panoptic segmentation model adaptors."""

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForImageSegmentation

from peekingduck.pipeline.nodes.model.huggingface_hubv1.adaptors import base


class PanopticSegmenter(base.HuggingFaceAdaptor):
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
