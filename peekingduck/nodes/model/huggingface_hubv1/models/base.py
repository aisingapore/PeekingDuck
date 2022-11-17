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

"""Base Hugging Face Hub model and adaptor."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from transformers import AutoConfig

from peekingduck.nodes.base import ThresholdCheckerMixin, WeightsDownloaderMixin
from peekingduck.nodes.model.huggingface_hubv1 import adaptors
from peekingduck.nodes.model.huggingface_hubv1.api_utils import get_valid_models


class HuggingFaceModel(ThresholdCheckerMixin, WeightsDownloaderMixin, ABC):
    """Abstract Hugging Face Hub model class."""

    adaptor: adaptors.base.HuggingFaceAdaptor
    _detect_ids: torch.Tensor

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.check_valid_choice("model_type", get_valid_models(self.config["task"]))  # type: ignore

        self.score_threshold = self.config["score_threshold"]

    @property
    def detect_ids(self) -> List[int]:
        """The list of selected object category IDs."""
        return self._detect_ids.int().tolist()

    @detect_ids.setter
    def detect_ids(self, ids: List[int]) -> None:
        if not isinstance(ids, list):
            raise TypeError("detect_ids has to be a list")
        self._detect_ids = torch.tensor(ids, device=self.device)

    @abstractmethod
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Predicts the supplied image.

        Args:
            image (np.ndarray): Input image frame.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains:
            - An array of detection bboxes
            - An array of human-friendly detection class names
            - An array of detection scores
            - An array of binary masks (if instance segmentation)

        Raises:
            TypeError: The provided `image` is not a numpy array.
        """

    def post_init(self) -> None:
        """Creates model and logs configuration after validation checks during
        initialization.
        """
        self._create_huggingface_model(self.config["model_type"])
        self._log_model_configs()

    @abstractmethod
    def _create_huggingface_model(self, model_type: str) -> None:
        """Creates the specified Hugging Face model from pretrained weights.
        In addition, also sets up cache directory paths, maps `detect` list to
        numeric detect IDs, and attempts to setup inference on GPU.

        Args:
            model_type (str): Hugging Face Hub model identifier
                (<organization>/<model name>), e.g., facebook/detr-resnet-50.
        """

    @abstractmethod
    def _log_model_configs(self) -> None:
        """Prints the loaded model's settings."""

    def _init_device(self) -> torch.device:
        """Initialize torch.device to `cuda` if available and then tries to
        move all inference-related attributes of the Hugging Face model to cuda.
        Reverts to Hugging Face's "cpu" default if moving fails.

        Returns:
            (torch.device): The device on which the tensors will be allocated.
        """
        device = torch.device(
            self.config["device"] if torch.cuda.is_available() else "cpu"
        )
        try:
            self.adaptor.to(device)
        except AttributeError:
            # Failed to move all the attributes to the same device, revert to
            # Hugging Face's default "cpu" device
            device = torch.device("cpu")

        return device

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
