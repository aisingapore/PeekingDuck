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

"""Performs multiple object tracking for detected bboxes."""

from typing import Any, Dict

from peekingduck.pipeline.nodes.dabble.trackingv1.detection_tracker import (
    DetectionTracker,
)
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """Uses bounding boxes detected by an object detector model to track
    multiple objects.

    Currently, two types of tracking algorithms can be selected: MOSSE and IOU.
    Information on the algorithms' performance can be found
    :ref:`here <object-tracking-benchmarks>`.

    Inputs:
        |img_data|

        |bboxes_data|

    Outputs:
        |obj_attrs_data|

    Configs:
        tracking_type (:obj:`str`): **{"iou", "mosse"}, default="iou"**. |br|
            Type of tracking algorithm to be used. For more information about
            the trackers, please view the :doc:`Multiple Object Tracking use
            case </use_cases/multiple_object_tracking>`.
        iou_threshold (:obj:`float`): **[0, 1], default=0.1**. |br|
            Minimum IoU value to be used with the matching logic.
        max_lost (:obj:`int`): **[0, sys.maxsize), default=10**. |br|
            Maximum number of frames to keep "lost" tracks after which they
            will be removed. Only used when ``tracking_type = iou``.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.tracker = DetectionTracker(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Tracks detection bounding boxes.

        Args:
            inputs (Dict[str, Any]): Dictionary with keys "img", "bboxes", and
                "bbox_scores.

        Returns:
            outputs (Dict[str, Any]): Tracking IDs of bounding boxes.
            "obj_attrs" key is used for compatibility with draw nodes.
        """
        # Potentially use frame_rate here too since IOUTracker has a
        # max_time_lost
        metadata = inputs.get("mot_metadata", {"reset_model": False})
        reset_model = metadata["reset_model"]
        if reset_model:
            self._reset_model()

        track_ids = self.tracker.track_detections(inputs)

        return {"obj_attrs": {"ids": track_ids}}

    def _reset_model(self) -> None:
        """Creates a new instance of DetectionTracker."""
        self.logger.info(f"Creating new {self.config['tracking_type']} tracker...")
        self.tracker = DetectionTracker(self.config)
