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
        config (Dict[str, Any]): Configration dict containing the following:
            tracking_type (str): Type of tracking algorithm to be used, one of
                ["iou", "mosse"].
            iou_threshold (float): Minimum IoU value to be used with the
                matching logic.
            max_lost (int): Maximum number of frames to keep "lost" tracks
                after which they will be removed. Only used in IOUTracker.

    Raises:
        ValueError: `tracking_type` is not one of ["iou", "mosse"].
        ValueError: `iou_threshold` is not within [0, 1].
        ValueError: `max_lost` is negative.
    """

    tracker_constructors = {"iou": IOUTracker, "mosse": OpenCVTracker}

    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = logging.getLogger(__name__)

        # Check threshold values
        if not 0 <= config["iou_threshold"] <= 1:
            raise ValueError("iou_threshold must be in [0, 1]")
        if config["max_lost"] < 0:
            raise ValueError("max_lost cannot be negative")

        try:
            self.tracker = self.tracker_constructors[config["tracking_type"]](config)
        except KeyError as error:
            raise ValueError("tracking_type must be one of ['iou', 'mosse']") from error

    def track_detections(self, inputs: Dict[str, Any]) -> List[int]:
        """Tracks detections using the selected algorithm.

        Args:
            inputs (Dict[str, Any]): Dictionary with keys "img", "bboxes", and
                "bbox_scores.

        Returns:
            (List[int]): Tracking IDs of the detection bounding boxes.
        """
        track_ids = self.tracker.track_detections(inputs)
        return track_ids
