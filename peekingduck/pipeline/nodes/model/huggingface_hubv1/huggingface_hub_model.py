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
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torchvision
from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForObjectDetection

from peekingduck.pipeline.nodes.base import (
    ThresholdCheckerMixin,
    WeightsDownloaderMixin,
)
from peekingduck.pipeline.nodes.model.huggingface_hubv1.api_utils import (
    get_valid_models,
)
from peekingduck.pipeline.utils.bbox.transforms import xyxy2xyxyn


class ObjectDetectionModel(  # pylint: disable=too-many-instance-attributes
    ThresholdCheckerMixin, WeightsDownloaderMixin
):
    """Validates configuration, loads model from Hugging Face Hub, and performs
    inference.

    Configuration options are validated to ensure they have valid types and
    values. Model weights files are downloaded if not found in the location
    indicated by the `weights_dir` configuration option.
    """

    _detect_ids: torch.Tensor

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.check_bounds(["iou_threshold", "score_threshold"], "[0, 1]")
        self.check_valid_choice("task", {"object_detection"})
        self.check_valid_choice("model_type", get_valid_models(self.config["task"]))  # type: ignore

        self.model_type = self.config["model_type"]
        self.iou_threshold = self.config["iou_threshold"]
        self.score_threshold = self.config["score_threshold"]
        self.agnostic_nms = self.config["agnostic_nms"]

        self._create_huggingface_model()
        self._log_model_configs()

    @property
    def detect_ids(self) -> List[int]:
        """The list of selected object category IDs."""
        return self._detect_ids.int().tolist()

    @detect_ids.setter
    def detect_ids(self, ids: List[int]) -> None:
        if not isinstance(ids, list):
            raise TypeError("detect_ids has to be a list")
        self._detect_ids = torch.tensor(ids, device=self.device)

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
        inputs = self.extractor(images=image, return_tensors="pt").to(self.device)
        outputs = self.detector(**inputs)

        target_sizes = torch.tensor([image.shape[:2]], device=self.device)
        result = self.extractor.post_process(
            outputs=outputs, target_sizes=target_sizes
        )[0]

        return self._postprocess(result, image_size)

    def _create_huggingface_model(self) -> None:
        """Creates the specified Hugging Face model from pretrained weights.
        In addition, also sets up cache directory paths, maps `detect` list to
        numeric detect IDs, and attempts to setup inference on GPU.
        """
        model_dir = self.prepare_cache_dir(self.config["model_type"].split("/"))

        detector_config = AutoConfig.from_pretrained(
            self.config["model_type"], cache_dir=model_dir
        )
        detect_ids = self._map_detect_ids(detector_config, self.config["detect"])

        self.extractor = AutoFeatureExtractor.from_pretrained(
            self.config["model_type"], cache_dir=model_dir
        )
        self.detector = AutoModelForObjectDetection.from_pretrained(
            self.config["model_type"], cache_dir=model_dir
        )
        self.device = self._init_device()
        self.detect_ids = detect_ids

    def _init_device(self) -> torch.device:
        """Initialize torch.device to `cuda` if available and then tries to
        move all inference-related attributes of the Hugging Face model to cuda.
        Reverts to Hugging Face's "cpu" default if moving fails.

        Returns:
            (torch.device): The device on which the tensors will be allocated.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(self.detector, "model"):
            self.detector.model.to(device)
        elif hasattr(self.detector, "vit"):
            self.detector.vit.to(device)
        else:
            # Failed to move all the attributes to the same device, revert to
            # Hugging Face's default "cpu" device
            device = torch.device("cpu")
        self.detector.class_labels_classifier.to(device)
        self.detector.bbox_predictor.to(device)
        return device

    def _log_model_configs(self) -> None:
        """Prints the loaded model's settings."""
        self.logger.info(
            "Hugging Face model loaded with the following configs:\n\t"
            f"Model type: {self.model_type}\n\t"
            f"IDs being detected: {self.detect_ids}\n\t"
            f"IOU threshold: {self.iou_threshold}\n\t"
            f"Score threshold: {self.score_threshold}\n\t"
            f"Class agnostic NMS: {self.agnostic_nms}"
        )

    def _map_detect_ids(  # pylint: disable=too-many-branches
        self, detector_config: AutoConfig, detect_list: List[Union[int, str]]
    ) -> List[int]:
        """Maps a list of detect IDs/labels to a list of valid numeric detect
        IDs.

        Defaults to detect all IDs if ``detect_list`` is either empty or only
        contains "*".

        Displays a warning if any invalid detect ID or label were used. If
        numeric detect ID was used in ``detect_list``, displays a warning to
        recommend users to use detect labels for consistent behavior.

        Args:
            detector_config (AutoConfig): A configuration class which contains
                label-to-id mapping and vice versa.
            detect_list (List[Union[int, str]]): A list containing numeric
                detect IDs and/or human-friendly class names.

        Returns:
            (List[int]): A list of numeric detect IDs.
        """
        detect_ids = []
        invalid_ids = []
        invalid_labels = []

        label_to_id = {
            label.lower(): id for label, id in detector_config.label2id.items()
        }
        if not detect_list:
            self.logger.warning("`detect` list is empty, detecting all objects.")
            detect_list = [*label_to_id]
        elif detect_list == ["*"]:
            self.logger.warning("Detecting all objects.")
            detect_list = [*label_to_id]

        use_numeric_id = False
        for detect_item in detect_list:
            if isinstance(detect_item, str):
                detect_label = detect_item.lower()
                if detect_label not in label_to_id:
                    invalid_labels.append(detect_item)
                else:
                    detect_ids.append(label_to_id[detect_label])
            else:
                use_numeric_id = True
                if detect_item not in detector_config.id2label:
                    invalid_ids.append(detect_item)
                else:
                    detect_ids.append(detect_item)

        if invalid_ids:
            self.logger.warning(f"Invalid detect IDs: {invalid_ids}")
        if invalid_labels:
            self.logger.warning(f"Invalid class names: {invalid_labels}")
        if use_numeric_id:
            self.logger.warning(
                "The models in this node have been train on various datasets. "
                "It is recommended to use class names instead of detect IDs "
                "for consistent behavior."
            )
        if not detect_ids:
            self.logger.warning(
                "No valid entries in `detect` list, detecting all objects."
            )
            detect_ids = [*detector_config.id2label]
        return sorted(list(set(detect_ids)))

    def _postprocess(
        self, result: Dict[str, torch.Tensor], image_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        classes = np.array(
            [self.detector.config.id2label[int(i)] for i in detections_np[:, 5]]
        )

        return bboxes, classes, scores
