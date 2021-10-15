# Modifications copyright 2021 AI Singapore

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Original copyright (c) 2017 TU Berlin, Communication Systems Group

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Create Tracker
"""

from collections import OrderedDict
import numpy as np
from scipy.spatial import distance
from .misc import get_centroid
from .track import Track


class Tracker:
    """
    Greedy Tracker with tracking based on ``centroid`` location of the
    bounding box of the object.

    Args:
        max_lost (int): Maximum number of consecutive frames object was not detected.
        tracker_output_format (str): Output format of the tracker.
    """

    def __init__(self, max_lost=5, tracker_output_format='mot_challenge'):
        self.next_track_id = 0
        self.tracks = OrderedDict()
        self.max_lost = max_lost
        self.frame_count = 0
        self.tracker_output_format = tracker_output_format

    def _add_track(self, frame_id, bbox, detection_confidence, class_id, **kwargs):
        """
        Add a newly detected object to the queue.

        Args:
            frame_id (int): Camera frame id.
            bbox (numpy.ndarray): Bounding box pixel coordinates as
                                  (xmin, ymin, xmax, ymax) of the track.
            detection_confidence (float): Detection confidence of the object (probability).
            class_id (str or int): Class label id.
            kwargs (dict): Additional key word arguments.
        """

        self.tracks[self.next_track_id] = Track(
            self.next_track_id, frame_id, bbox, detection_confidence,
            class_id=class_id, data_output_format=self.tracker_output_format,
            **kwargs
        )
        self.next_track_id += 1

    def _remove_track(self, track_id):
        """
        Remove tracker data after object is lost.

        Args:
            track_id (int): track_id of the track lost while tracking.
        """

        del self.tracks[track_id]

    # pylint: disable=too-many-arguments
    def _update_track(self, track_id, frame_id, bbox, detection_confidence,
                      class_id, lost=0, iou_score=0., **kwargs):
        """
        Update track state.

        Args:
            track_id (int): ID of the track.
            frame_id (int): Frame count.
            bbox (numpy.ndarray or list): Bounding box coordinates as
                                          `(xmin, ymin, width, height)`.
            detection_confidence (float): Detection confidence (a.k.a. detection probability).
            class_id (int): ID of the class (aka label) of the object being tracked.
            lost (int): Number of frames the object was lost while tracking.
            iou_score (float): Intersection over union.
            kwargs (dict): Additional keyword arguments.
        """

        self.tracks[track_id].update(
            frame_id, bbox, detection_confidence, class_id=class_id, lost=lost,
            iou_score=iou_score, **kwargs
        )

    @staticmethod
    def _get_tracks(tracks):
        """
        Output the information of tracks.

        Args:
            tracks (OrderedDict): Tracks dictionary with (key, value) as
                                  (track_id, corresponding `Track` objects).

        Returns:
            list: List of tracks being currently tracked by the tracker.
        """

        outputs = []
        for _, track in tracks.items():
            if not track.lost:
                outputs.append(track.output())
        return outputs

    @staticmethod
    def preprocess_input(bboxes, class_ids, detection_scores):
        """
        Preprocess the input data.

        Args:
            bboxes (list or numpy.ndarray): Array of bounding boxes with
                each bbox as a tuple containing `(xmin, ymin, width, height)`.
            class_ids (list or numpy.ndarray): Array of Class ID or label ID.
            detection_scores (list or numpy.ndarray): Array of detection
                scores (a.k.a. detection probabilities).

        Returns:
            detections (list[Tuple]): Data for detections as list of tuples
                containing `(bbox, class_id, detection_score)`.
        """

        new_bboxes = np.array(bboxes, dtype='int')
        new_class_ids = np.array(class_ids, dtype='int')
        new_detection_scores = np.array(detection_scores)

        new_detections = list(zip(new_bboxes, new_class_ids, new_detection_scores))
        return new_detections

    # pylint: disable=too-many-locals
    def update(self, bboxes, detection_scores, class_ids):
        """
        Update the tracker based on the new bounding boxes.

        Args:
            bboxes (numpy.ndarray or list): List of bounding boxes detected
                in the current frame. Each element of the list represent
                coordinates of bounding box as tuple
                `(top-left-x, top-left-y, width, height)`.
            detection_scores(numpy.ndarray or list): List of detection
                scores (probability) of each detected object.
            class_ids (numpy.ndarray or list): List of class_ids (int)
                corresponding to labels of the detected object. Default is `None`.

        Returns:
            list: List of tracks being currently tracked by the tracker.
                Each track is represented by the tuple with elements
                `(frame_id, track_id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z)`.
        """

        self.frame_count += 1

        if len(bboxes) == 0:
            lost_ids = list(self.tracks.keys())

            for track_id in lost_ids:
                self.tracks[track_id].lost += 1
                if self.tracks[track_id].lost > self.max_lost:
                    self._remove_track(track_id)

            outputs = self._get_tracks(self.tracks)
            return outputs

        detections = Tracker.preprocess_input(bboxes, class_ids, detection_scores)

        track_ids = list(self.tracks.keys())

        updated_tracks, updated_detections = [], []

        # pylint: disable=len-as-condition
        if len(track_ids):
            track_centroids = np.array([self.tracks[tid].centroid for tid in track_ids])
            detection_centroids = get_centroid(bboxes)

            centroid_distances = distance.cdist(track_centroids, detection_centroids)

            track_indices = np.amin(centroid_distances, axis=1).argsort()

            for idx in track_indices:
                track_id = track_ids[idx]

                remaining_detections = [
                    (i, d) for (i, d) in enumerate(centroid_distances[idx, :]) \
                    if i not in updated_detections]

                # pylint: disable=len-as-condition
                if len(remaining_detections):
                    detection_idx, _ = min(remaining_detections,
                                                            key=lambda x: x[1])
                    bbox, class_id, confidence = detections[detection_idx]
                    self._update_track(track_id, self.frame_count, bbox, confidence,
                                       class_id=class_id)
                    updated_detections.append(detection_idx)
                    updated_tracks.append(track_id)

                if len(updated_tracks) == 0 or track_id is not updated_tracks[-1]:
                    self.tracks[track_id].lost += 1
                    if self.tracks[track_id].lost > self.max_lost:
                        self._remove_track(track_id)

        for i, (bbox, class_id, confidence) in enumerate(detections):
            if i not in updated_detections:
                self._add_track(self.frame_count, bbox, confidence, class_id=class_id)

        outputs = self._get_tracks(self.tracks)
        return outputs
