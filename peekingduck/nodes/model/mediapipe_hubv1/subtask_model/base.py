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

"""Base subtask model."""


from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np

from peekingduck.nodes.base import ThresholdCheckerMixin
from peekingduck.utils.abstract_class_attributes import abstract_class_attributes


class ModelSetting(NamedTuple):
    """A single model setting and its corresponding names when used for
    instantiating MediaPipe models and logging.
    """

    # Keyword argument name
    keyword: Optional[str]
    # Logging option name
    option: str
    # Config value
    value: Any


@abstract_class_attributes("KEYPOINT_FORMATS")
class BaseEstimator(ThresholdCheckerMixin, ABC):
    """Base class to handle MediaPipe pose estimation subtasks."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

        self.check_valid_choice("keypoint_format", self.KEYPOINT_FORMATS)
        self.settings = [
            ModelSetting("model_complexity", "Model type", config["model_type"]),
            ModelSetting(None, "Keypoint format", config["keypoint_format"]),
            ModelSetting(
                "min_detection_confidence", "Score threshold", config["score_threshold"]
            ),
            ModelSetting(
                "min_tracking_confidence",
                "Tracking score threshold",
                config["tracking_score_threshold"],
            ),
            ModelSetting(
                "static_image_mode", "Static image mode", config["static_image_mode"]
            ),
        ]

    @property
    def arguments(self) -> Dict[str, Any]:
        """A dictionary of keyword arguments and their respective values."""
        return {
            setting.keyword: setting.value
            for setting in self.settings
            if setting.keyword is not None
        }

    @property
    def loaded_config(self) -> str:
        """A string of the loaded config options and their respective values for logging."""
        return "\n\t".join(
            f"{setting.option}: {setting.value}" for setting in self.settings
        )

    @abstractmethod
    def postprocess(self, result: Any) -> Tuple[np.ndarray, ...]:
        """Post processes detection result. Converts the bounding boxes from
        normalized [t, l, w, h] to normalized [x1, y1, x2, y2] format. Manually
        creates a detection label for each detection.

        Args:
            result (Any): Pose estimation results which consists of landmark
                coordinates and visibility scores.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains:
            - An array of detection bboxes
            - An array of human-friendly detection class names
            - An array of keypoint coordinates
            - An array of keypoint connections
            - An array of keypoint scores
        """

    def process(self, image: np.ndarray) -> Any:
        """Wrapper for `model`'s process() method."""
        return self.model.process(image)
