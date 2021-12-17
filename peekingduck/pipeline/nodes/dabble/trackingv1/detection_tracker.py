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

"""Tracker for object detector bounding boxes."""

import logging
from typing import Any, Dict, List

from peekingduck.pipeline.nodes.dabble.trackingv1.tracking_files.iou_tracker import (
    IOUTracker,
)
from peekingduck.pipeline.nodes.dabble.trackingv1.tracking_files.opencv_tracker import (
    OpenCVTracker,
)


class DetectionTracker:  # pylint: disable=too-few-public-methods
    """Tracks detection bounding boxes using the chosen algorithm.

    Args:
        tracker_type (str): Type of tracking algorithm to be used, one of
            ["iou", "mosse"].

    Raises:
        ValueError: `tracker_type` is not one of ["iou", "mosse"].
    """

    tracker_constructors = {"iou": IOUTracker, "mosse": OpenCVTracker}

    def __init__(self, tracker_type: str) -> None:
        self.logger = logging.getLogger(__name__)

        try:
            self.tracker = self.tracker_constructors[tracker_type]()
        except KeyError as error:
            raise ValueError("tracker_type must be one of ['iou', 'mosse']") from error

    def track_detections(self, inputs: Dict[str, Any]) -> List[str]:
        """Tracks detections using the selected algorithm.

        Args:
            inputs (Dict[str, Any]): Dictionary with keys "img", "bboxes", and
                "bbox_scores.

        Returns:
            (List[str]): Tracking IDs of the detection bounding boxes.
        """
        track_ids = self.tracker.track_detections(inputs)
        return track_ids
