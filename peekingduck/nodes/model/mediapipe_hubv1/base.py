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

"""Base MediaPipe model wrapper."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Set, Tuple

import numpy as np
from mediapipe.python.solution_base import (  # pylint: disable=no-name-in-module
    SolutionBase,
)

from peekingduck.nodes.base import ThresholdCheckerMixin
from peekingduck.utils.abstract_class_attributes import abstract_class_attributes


@abstract_class_attributes("SUBTASK_MODEL_TYPES", "SUBTASKS")
class MediaPipeModel(ThresholdCheckerMixin, ABC):
    """Base MediaPipe model with abstract methods and attributes to be
    instantiated by sub classes.
    """

    model: SolutionBase

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.check_valid_choice("subtask", self.SUBTASKS)  # type: ignore
        self.check_valid_choice("model_type", self.model_types)  # type: ignore
        self.check_bounds("score_threshold", "[0, 1]")

    @property
    @abstractmethod
    def model_settings(self) -> Dict[str, Any]:
        """Model settings to be used with various MediaPipe model constructors."""

    @property
    def model_types(self) -> Set[int]:
        """Model types for the selected subtask."""
        return self.SUBTASK_MODEL_TYPES[self.subtask]

    @property
    def subtask(self) -> str:
        """Returns the current configured subtask."""
        return self.config["subtask"]

    def post_init(self) -> None:
        """Creates model and logs configuration after validation checks during
        initialization.
        """
        self._create_mediapipe_model(self.model_settings)

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Performs inference on the provided image.

        Args:
            image (np.ndarray): Input image frame.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains:
            - An array of detection bboxes
            - An array of detection labels
            - An array of detection scores (object detection only)
            - An array of keypoint coordinates (pose estimation only)
            - An array of keypoint connections (pose estimation only)
            - An array of keypoint scores (pose estimation only)

        Raises:
            TypeError: The provided `image` is not a numpy array.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a np.ndarray")

        result = self.model.process(image)
        return self._postprocess(result)

    @abstractmethod
    def _create_mediapipe_model(self, model_settings: Dict[str, Any]) -> None:
        """Creates the MediaPipe model and logs the settings used."""

    @abstractmethod
    def _postprocess(self, result: Any) -> Tuple[np.ndarray, ...]:
        """Post processes detection result. Converts PeekingDuck compatible
        format.

        Args:
            result (Any): Inference output from MediaPipe models.

        Returns:
            (Tuple[np.ndarray, ...]): Returned tuple
            contains:
            - An array of detection bboxes
            - An array of human-friendly detection class names
            - An array of bbox scores (object detection only)
            - An array of keypoint coordinates (pose estimation only)
            - An array of keypoint connections (pose estimation only)
            - An array of keypoint scores (pose estimation only)
        """
