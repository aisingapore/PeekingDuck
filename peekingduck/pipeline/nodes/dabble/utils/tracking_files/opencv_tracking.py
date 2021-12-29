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
Tracking algorithm that uses OpenCV's MOSSE.
"""

from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np

from peekingduck.pipeline.nodes.dabble.utils.tracking_files.iou_tracker.utils import (
    format_boxes,
    iou,
)


class OpenCVTracker:  # pylint: disable=too-few-public-methods
    """Native OpenCV tracker that is initialized on bounding boxes detected
    in first frame of video feed.

    Only the "MOSSE" tracker can be selected as it operates at a high FPS.

    References:
        Inference code adapted from:
            https://learnopencv.com/object-tracking-using-opencv-cpp-python/
    """

    def __init__(self) -> None:
        super().__init__()
        self.tracker_type = "MOSSE"
        self.first_frame_or_not = True
        self.next_object_id = 0
        self.iou_thresh = 0.1
        # Dict to store {id (key): [Tracker, bbox(prev)]}
        self.tracking_dict: Dict[int, List[Any]] = {}

    def run(self, inputs: Dict[str, Any]) -> List[str]:
        """Initialises and update tracker on each frame.

        Args:
            inputs (Dict[str, Any]): Outputs from previous nodes used.

        Returns:
            List[str]: List of track_ids sorted by bounding boxes.
        """
        frame = np.copy(inputs["img"])
        original_h, original_w, _ = frame.shape
        # Format bboxes from normalized to frame axis
        bboxes = np.copy(inputs["bboxes"])
        bboxes = format_boxes(bboxes, original_h, original_w)
        track_id = []

        # Read the first frame and initialize trackers. The single object
        # tracker is initialized using the first frame and the bounding
        # box indicating the location of the object we want to track
        if self.first_frame_or_not:
            for bbox in bboxes:
                self._initialise_tracker(bbox, frame)
            track_id = list(self.tracking_dict.keys())
            obj_tags = [str(x) for x in track_id]
            self.first_frame_or_not = False
        # Continuous frames
        else:
            obj_tags = self._if_new_bbox_add_track(bboxes, frame)

        # Get updated location of objects in subsequent frames
        for id_num, tracker in self.tracking_dict.copy().items():
            success, bbox = tracker[0].update(frame)
            if success:
                # update bounding box
                self.tracking_dict.update({id_num: [tracker[0], bbox]})
            else:
                del self.tracking_dict[id_num]

        return obj_tags

    def _if_new_bbox_add_track(
        self, bboxes: np.ndarray, frame: np.ndarray
    ) -> List[str]:
        """Checks for new bboxes added and initialises new tracker.

        Args:
            bboxes (np.ndarray): Detected bounding boxes.
            frame (np.ndarray): Image frame parsed from video.

        Returns:
            List[str]: List of track_ids.
        """
        prev_frame_tracked_bbox = []
        # Dict to store {current frame bbox: highest_iou_index}
        matching_dict: Dict[Tuple[float, ...], Any] = {}

        # Get previous frames' tracked bboxes
        for _, value in self.tracking_dict.items():
            prev_frame_tracked_bbox.append(np.array(value[1]))

        for box in bboxes:
            # Get matching ious for each bbox in frame to previous bboxes
            ious = iou(np.array(box), np.array(prev_frame_tracked_bbox))
            # Check if current bbox passes iou_thresh with any previous
            # tracked bboxes and get index of highest iou above threshold
            prev_frame_bbox_highest_iou_index = (
                ious.argmax() if round(max(ious), 1) >= self.iou_thresh else None
            )
            matching_dict.update({tuple(box): prev_frame_bbox_highest_iou_index})

        # Create object tags from highest IOU index and tracking_dict
        track_id = []
        for key, value in matching_dict.items():
            if value is not None:
                # Get object ID through prev_frame_bbox_highest_iou_index
                id_num = list(self.tracking_dict)[value]  # type: ignore
                track_id.append(str(id_num))
            else:
                # Create new tracker for bbox that < IOU threshold
                self._initialise_tracker(key, frame)
                id_num = list(self.tracking_dict)[-1]
                track_id.append(str(id_num))

        # Create result list to replace duplicate track_ids
        obj_tags = []
        for id_num in track_id:
            if id_num not in obj_tags:
                obj_tags.append(id_num)
            else:
                obj_tags.append("")

        return obj_tags

    def _initialise_tracker(
        self, bbox: Union[np.ndarray, Tuple[float, ...]], frame: np.ndarray
    ) -> Dict[int, List[Any]]:
        """Starts a tracker for each bbox.

        Args:
            bbox (Union[np.ndarray, Tuple[float, ...]]): Single detected
                bounding box.
            frame (np.ndarray): Image frame parsed from video.

        Returns:
            Dict[int, List[Any]]: Dict to store {id (key): [Tracker, bbox(prev)]}
        """
        tracker = self._create_tracker_by_name(self.tracker_type)
        tracker.init(frame, tuple(bbox))
        self.next_object_id += 1
        self.tracking_dict.update({self.next_object_id: [tracker, bbox]})

        return self.tracking_dict

    @staticmethod
    def _create_tracker_by_name(tracker_type: str) -> Any:
        """Create tracker based on tracker name.

        Args:
            tracker_type (str): Type of opencv tracker used.

        Returns:
            Any: MOSSE Tracker.
        """
        if tracker_type == "MOSSE":
            tracker = cv2.legacy.TrackerMOSSE_create()
        else:
            raise ValueError("Incorrect tracker type.")

        return tracker
