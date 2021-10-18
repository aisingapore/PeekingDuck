# Copyright 2021 AI Singapore
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

"""
Load Tracker for inference
"""

from typing import Any, Dict, List
import logging
from .tracking_files.iou_tracking import IOUTracking
from .tracking_files.opencv_tracking import OpenCVTracker


class LoadTracker:  # pylint: disable=too-few-public-methods
    """Loads chosen tracker node."""
    def __init__(self, tracking_type: str) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        if tracking_type == "mosse":
            self.logger.info('OpenCV Tracking algorithm used: %s', tracking_type)
            self.tracker = OpenCVTracker()  # type: ignore
        elif tracking_type == "iou":
            self.logger.info('Tracking algorithm used: %s', tracking_type)
            self.tracker = IOUTracking()  # type: ignore
        else:
            raise ValueError("tracking_type must be one of ['iou', 'mosse']")

    def run(self, inputs: Dict[str, Any]) -> List[str]:
        """Run tracking algorithm"""
        obj_tags = self.tracker.run(inputs)

        return obj_tags
