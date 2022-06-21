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

"""Tracking-by-detection using OpenCV's MOSSE Tracker."""

from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np

from peekingduck.pipeline.nodes.dabble.trackingv1.tracking_files.utils import (
    iou_candidates,
    xyxyn2tlwh,
)


class OpenCVTracker:  # pylint: disable=too-few-public-methods
    """OpenCV's MOSSE tracker.

    Attributes:
        is_initialized (bool): Flag to determine this is the first run of the
            tracker.
        iou_threshold (float): Minimum IoU value to be used with the matching
            logic.
        next_track_id (int): ID for the next (unmatched) object detection to be
            tracked.
        tracks (Dict[int, Track]): Maps track IDs to their respective
            tracker and bbox coordinates.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.is_initialized = False
        self.iou_threshold = config["iou_threshold"]
        self.next_track_id = 0
        self.tracks: Dict[int, Track] = {}

    def track_detections(self, inputs: Dict[str, Any]) -> List[int]:
        """Initializes and updates the tracker on each frame.

        Args:
            inputs (Dict[str, Any]): Dictionary with keys "img" and "boxes".

        Returns:
            (List[int]): Tracking IDs of the detection bounding boxes.
        """
        frame = inputs["img"]
        frame_size = frame.shape[:2]
        tlwhs = xyxyn2tlwh(inputs["bboxes"], *frame_size)

        if self.is_initialized:
            obj_track_ids = self._match_and_track(frame, tlwhs)
        else:
            for tlwh in tlwhs:
                self._initialize_tracker(frame, tlwh)
            obj_track_ids = list(self.tracks.keys())
        self._update_tracker_bboxes(frame)

        return obj_track_ids

    def _initialize_tracker(self, frame: np.ndarray, bbox: np.ndarray) -> None:
        """Starts a tracker for each bbox.

        Args:
            frame (np.ndarray): Image frame parsed from video.
            bbox (np.ndarray): Single detected bounding box.
        """
        tracker = cv2.legacy.TrackerMOSSE_create()
        tracker.init(frame, tuple(bbox))
        self.tracks[self.next_track_id] = Track(tracker, bbox)
        self.next_track_id += 1
        self.is_initialized = True

    def _match_and_track(self, frame: np.ndarray, bboxes: np.ndarray) -> List[int]:
        """Matches detections to tracked bboxes, creates a new track if no
        match is found.

        Args:
            frame (np.ndarray): Input video frame.
            bboxes (np.ndarray): Detection bounding boxes with (t, l, w, h)
                format where (t, l) is the coordinate of the top-left corner,
                w is the width, and h is the height of the bounding box
                respectively.

        Returns:
            (List[int]): A list of track IDs for the detections in the current
                frame.
        """
        prev_tracks = [track.bbox for _, track in self.tracks.items()]
        prev_tracked_bboxes = np.array(prev_tracks) if prev_tracks else np.empty((0, 4))
        matching_dict: Dict[Tuple[float, ...], Optional[np.intp]] = {}

        for bbox in bboxes:
            ious = iou_candidates(bbox, prev_tracked_bboxes)
            if len(ious) > 0 and max(ious) >= self.iou_threshold:
                matching_dict[tuple(bbox)] = ious.argmax()
            else:
                matching_dict[tuple(bbox)] = None

        track_ids = []
        for bbox, matched_id in matching_dict.items():
            if matched_id is None:
                self._initialize_tracker(frame, np.array(bbox))
                track_ids.append(list(self.tracks)[-1])
            else:
                track_ids.append(list(self.tracks)[matched_id])

        return track_ids

    def _update_tracker_bboxes(self, frame: np.ndarray) -> None:
        """Updates location of previously tracked detections in subsequent
        frames. Removes the track if its tracker fails to update.
        """
        failures = []
        for track_id, track in self.tracks.items():
            success, bbox = track.tracker.update(frame)
            if success:
                self.tracks[track_id] = Track(track.tracker, np.array(bbox))
            else:
                failures.append(track_id)
        for track_id in failures:
            del self.tracks[track_id]


class Track(NamedTuple):
    """Stores the individual OpenCV MOSSE tracker and its bounding box
    coordinates.

    Attributes:
        tracker (cv2.legacy_TrackerMOSSE): OpenCV MOSSE tracker.
        bbox (np.ndarray): Bounding box coordinates in (t, l, w, h) format where
            (t, l) is the top-left coordinate, w is the width, and h is the
            height of the bounding box.
    """

    tracker: cv2.legacy_TrackerMOSSE
    bbox: np.ndarray
