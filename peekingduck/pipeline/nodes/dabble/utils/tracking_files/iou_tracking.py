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
Tracking algorithm that uses IOU matching.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from .iou_tracker.iou_tracker import IOUTracker
from .iou_tracker.utils import format_boxes


class IOUTracking:  # pylint: disable=too-few-public-methods
    """Simple tracking class based on Intersection Over Union of bounding
    boxes.

    This method is based on the assumption that the detector produces a
    detection per frame for every object to be tracked. Furthermore, it
    is assumed that detections of an object in consecutive frames have
    an unmistakably high overlap IOU (intersection-over-union) which is
    commonly the case when using sufficiently high frame rates.

    References:
        High-Speed Tracking-by-Detection Without Using Image Information:
            http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf

        Inference code adapted from https://github.com/adipandas/multi-object-tracker
    """

    def __init__(self) -> None:
        super().__init__()
        self.tracker = IOUTracker(
            max_lost=10,
            iou_threshold=0.1,
            min_detection_confidence=0.2,
            max_detection_confidence=1,
        )

    def run(self, inputs: Dict[str, Any]) -> List[str]:
        """Updates tracker on each frame and return sorted object tags.

        Args:
            inputs (Dict[str, Any]): Outputs from previous nodes used.

        Returns:
            List[str]: List of track_ids sorted by bounding boxes.
        """

        frame = np.copy(inputs["img"])
        original_h, original_w, _ = frame.shape
        bboxes = np.copy(inputs["bboxes"])

        # Format bboxes from normalized to frame axis
        bboxes = format_boxes(bboxes, original_h, original_w)
        confidences = np.copy(inputs["bbox_scores"])
        class_ids = self._convert_class_label_to_unique_id(
            np.copy(inputs["bbox_labels"])
        )

        # Update trackers with current bboxes and scores
        tracks = self.tracker.update(bboxes, confidences, class_ids)
        # Order object tags by current bbox order
        obj_tags = self._order_tags_by_bbox(bboxes, tracks)

        return obj_tags

    @staticmethod
    def _order_tags_by_bbox(bboxes: np.ndarray, tracks: List[Any]) -> List[str]:
        """Orders object tags by bboxes.

        Args:
            bboxes (np.ndarray): Detected bounding boxes.
            tracks (List[Any]): List of tracks.

        Returns:
            List[str]: List of track_ids sorted by bounding boxes.
        """

        # Create dict with {(bbox): id}
        matching_dict: Dict[Tuple[Any, ...], Any] = {}
        for trk in tracks:
            matching_dict.update({(trk[2], trk[3], trk[4], trk[5]): trk[1]})

        obj_tags = []
        # Match current bbox with dict key and return id (value)
        for bbox in bboxes:
            obj_tag = matching_dict[tuple(bbox)]
            obj_tags.append(str(obj_tag))

        return obj_tags

    @staticmethod
    def _convert_class_label_to_unique_id(classes: List[str]) -> List[int]:
        """Converts class label to unique IDs.

        Args:
            classes (List[str]): Bounding box class labels.

        Returns:
            List[int]: Unique ID per label.
        """

        obj_id = {}
        for num, label in enumerate(sorted(set(classes))):
            obj_id.update({label: num + 1})
        class_ids = [obj_id[x] for x in classes]

        return class_ids
