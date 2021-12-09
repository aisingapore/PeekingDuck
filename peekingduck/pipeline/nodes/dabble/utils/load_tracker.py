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
Load Tracker for inference.
"""

from typing import Any, Dict, List
import logging
from peekingduck.pipeline.nodes.dabble.utils.tracking_files.iou_tracking import (
    IOUTracking,
)
from peekingduck.pipeline.nodes.dabble.utils.tracking_files.opencv_tracking import (
    OpenCVTracker,
)


class TrackerLoader:  # pylint: disable=too-few-public-methods
    """Loads chosen tracker node."""

    def __init__(self, tracking_type: str) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Tracking algorithm used: {tracking_type}")
        if tracking_type in ["iou", "mosse"]:
            self.tracker = self._get_tracker(tracking_type)
        else:
            raise ValueError("tracking_type must be one of ['iou', 'mosse']")

    def predict(self, inputs: Dict[str, Any]) -> List[str]:
        """Runs tracking algorithm.

        Args:
            inputs (Dict[str, Any]): Dict of outputs from earlier nodes.

        Returns:
            List[str]: Tracking ids of bounding boxes and ordered
                accordingly.
        """
        obj_tags = self.tracker.run(inputs)

        return obj_tags

    @staticmethod
    def _get_tracker(tracking_type: str) -> Any:
        """Returns tracker from tracking_type config parameter.

        Args:
            tracking_type (str): Name of tracker to be used.

        Returns:
            Any: Tracker type.
        """
        trackers_dict = {"iou": IOUTracking(), "mosse": OpenCVTracker()}
        tracker = trackers_dict[tracking_type]
        return tracker
