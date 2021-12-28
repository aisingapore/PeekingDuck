# Modifications copyright 2021 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Original copyright (c) 2017 TU Berlin, Communication Systems Group
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Tracking-by-detection using IoU Tracker."""

from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import numpy as np

from peekingduck.pipeline.nodes.dabble.trackingv1.tracking_files.track import Track
from peekingduck.pipeline.nodes.dabble.trackingv1.tracking_files.utils import (
    iou_tlwh,
    xyxyn2tlwh,
)


class IOUTracker:
    """Simple tracking class based on Intersection over Union (IoU) of bounding
    boxes.

    This method is based on the assumption that the detector produces a
    detection per frame for every object to be tracked. Furthermore, it is
    assumed that detections of an object in consecutive frames have an
    unmistakably high overlap IoU which is commonly the case when using
    sufficiently high frame rates.

    References:
        High-Speed Tracking-by-Detection Without Using Image Information:
        http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf

        Inference code adapted from
        https://github.com/adipandas/multi-object-tracker
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.iou_threshold = config["iou_threshold"]
        self.max_lost = config["max_lost"]
        self.next_track_id = 0

        self.tracks: "OrderedDict[int, Track]" = OrderedDict()

    def track_detections(self, inputs: Dict[str, Any]) -> List[str]:
        """Initialises and updates tracker on each frame.

        Args:
            inputs (Dict[str, Any]): Dictionary with keys "img" and "bboxes".

        Returns:
            (List[str]): List of track IDs.
        """
        frame = inputs["img"]
        frame_size = frame.shape[:2]
        tlwhs = xyxyn2tlwh(inputs["bboxes"], *frame_size)

        tracks = self.update(list(tlwhs))
        track_ids = self._order_track_ids_by_bbox(tlwhs, tracks)

        return track_ids

    def update(self, detections: np.ndarray) -> List[Track]:
        """Updates the tracker. Creates new tracks for untracked objects,
        updates tracked objects with the new class ID and bounding box
        coordinates. Removes tracks which have not been detected for longer
        than the `max_lost` threshold.

        Args:
            detections (np.ndarray): Bounding box coordinates for each of
                detection. The bounding box has the format (t, l, w, h) where
                (t, l) is the top-left corner, w is the width, and h is the
                height.

        Returns:
            (List[Track]): All tracked detections in the current frame.
        """
        track_ids = list(self.tracks.keys())

        updated_tracks = []
        for track_id in track_ids:
            if detections:
                idx, best_match, best_iou = self.get_best_match_by_iou(
                    detections, self.tracks[track_id].bbox
                )
                if best_iou >= self.iou_threshold:
                    self._update_track(track_id, best_match, best_iou)
                    updated_tracks.append(track_id)
                    del detections[idx]
            if not updated_tracks or track_id != updated_tracks[-1]:
                self.tracks[track_id].lost += 1
                if self.tracks[track_id].lost > self.max_lost:
                    self._remove_track(track_id)
        for tlwh in detections:
            self._add_track(tlwh)
        outputs = self._get_tracks()
        return outputs

    def _add_track(self, bbox: np.ndarray) -> None:
        """Adds a newly detected object to the list of tracked detections.

        Args:
            bbox (np.ndarray): Bounding box with format (t, l, w, h) where
                (t, l) is the top-left corner, w is the width, and h is the
                height.
        """
        self.tracks[self.next_track_id] = Track(self.next_track_id, bbox)
        self.next_track_id += 1

    def _get_tracks(self) -> List[Track]:
        """All tracked detections in the current frame."""
        return [track for _, track in self.tracks.items() if track.lost == 0]

    def _remove_track(self, track_id: int) -> None:
        """Removes the specified track. Typically called when the track has not
        been detected in the frame for longer than `max_lost` consecutive
        frames.

        Args:
            track_id (int): The track to delete.
        """
        del self.tracks[track_id]

    def _update_track(self, track_id: int, bbox: np.ndarray, iou_score: float) -> None:
        """Updates the specified tracked detection.

        Args:
            track_id (int): ID of the tracked detection.
            bbox (np.ndarray): Bounding box coordinates with (t, l, w, h)
                format where (t, l) is the top-left corner, w is the width, and
                h is the height.
            iou_score (float): Intersection-over-Union between the current
                detection bounding box and its last detected bounding box.
        """
        self.tracks[track_id].update(bbox, iou_score)

    @staticmethod
    def get_best_match_by_iou(
        detections: List[np.ndarray], tracked_bbox: np.ndarray
    ) -> Tuple[int, np.ndarray, float]:
        """Finds the best match between all the current detections and the
        specified tracked bounding box. Best match is the pair with the largest
        IoU value.

        Args:
            detections (List[np.ndarray]): List of tuples containing the
                bounding box coordinates for each of the detection. The
                bounding box has the format (t, l, w, h) where (t, l) is the
                top-left corner, w is the width, and h is the height.
            tracked_bbox (np.ndarray): The specified tracked bounding box.

        Returns:
            (Tuple[int, np.ndarray, float]): The index, bounding box of the
            best match current detection, and the IoU value.
        """
        detection_and_iou = lambda det: (det, iou_tlwh(tracked_bbox, det))
        idx, (best_match, best_iou) = max(
            enumerate(map(detection_and_iou, detections)),
            key=lambda x: x[1][1],
        )
        return idx, best_match, best_iou

    @staticmethod
    def _order_track_ids_by_bbox(bboxes: np.ndarray, tracks: List[Track]) -> List[str]:
        """Extracts the track IDs and orders them by their respective bounding
        boxes.

        Args:
            bboxes (np.ndarray): Detection bounding boxes with (t, l, w, h)
                format where (t, l) is the top-left corner, w is the width, and
                h is the height.
            tracks (List[Track]): List of tracked detections.

        Returns:
            (List[str]): Track IDs of the detections in the current frame.
        """
        bbox_to_track_id = {tuple(track.bbox): track.track_id for track in tracks}
        track_ids = [str(bbox_to_track_id[bbox]) for bbox in map(tuple, bboxes)]
        return track_ids
