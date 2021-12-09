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

"""
IOU Tracker.
"""

from typing import Any, List
import numpy as np
from peekingduck.pipeline.nodes.dabble.utils.tracking_files.iou_tracker.tracker import (
    Tracker,
)
from peekingduck.pipeline.nodes.dabble.utils.tracking_files.iou_tracker.misc import (
    iou_xywh as iou,
)


class IOUTracker(Tracker):  # pylint: disable=too-few-public-methods
    """Intersection over Union Tracker.

    Args:
        max_lost (int): Maximum number of consecutive frames object was not detected.
        tracker_output_format (str): Output format of the tracker. Default set to
            'mot_challenge'.
        min_detection_confidence (float): Threshold for minimum detection confidence.
        max_detection_confidence (float): Threshold for maximum detection confidence.
        iou_threshold (float): Intersection over union minimum value.

    References:
        Implementation of this algorithm is heavily based on:
            https://github.com/bochinski/iou-tracker
    """

    def __init__(
        self,
        max_lost: int = 2,
        iou_threshold: float = 0.5,
        min_detection_confidence: float = 0.4,
        max_detection_confidence: float = 0.7,
    ) -> None:
        super().__init__(max_lost, tracker_output_format="mot_challenge")
        self.iou_threshold = iou_threshold
        self.max_detection_confidence = max_detection_confidence
        self.min_detection_confidence = min_detection_confidence

    # pylint: disable=too-many-locals
    def update(
        self, bboxes: np.ndarray, detection_scores: List[float], class_ids: List[int]
    ) -> List[Any]:
        detections = Tracker.preprocess_input(bboxes, class_ids, detection_scores)
        self.frame_count += 1
        track_ids = list(self.tracks.keys())

        updated_tracks = []
        # pylint: disable=cell-var-from-loop
        for track_id in track_ids:
            if detections:
                idx, best_match = max(
                    enumerate(detections),
                    key=lambda x: iou(self.tracks[track_id].bbox, x[1][0]),
                )
                (box, cid, scr) = best_match

                if iou(self.tracks[track_id].bbox, box) > self.iou_threshold:
                    self._update_track(
                        track_id,
                        self.frame_count,
                        box,
                        scr,
                        class_id=cid,
                        iou_score=iou(self.tracks[track_id].bbox, box),
                    )
                    updated_tracks.append(track_id)
                    del detections[idx]

            if not updated_tracks or track_id != updated_tracks[-1]:
                self.tracks[track_id].lost += 1
                if self.tracks[track_id].lost > self.max_lost:
                    self._remove_track(track_id)

        for bbox, class_id, score in detections:
            self._add_track(self.frame_count, bbox, score, class_id)

        outputs = self._get_tracks(self.tracks)
        return outputs
