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
Track class
"""

from typing import Any, Dict, Tuple, Union
import numpy as np

class Track:  # pylint: disable=too-many-instance-attributes
    """
    Track containing attributes to track various objects.

    Args:
        frame_id (int): Camera frame id.
        track_id (int): Track Id
        bbox (numpy.ndarray): Bounding box pixel coordinates as
            (xmin, ymin, width, height) of the track.
        detection_confidence (float): Detection confidence of the object (probability).
        class_id (str or int): Class label id.
        lost (int): Number of times the object or track was not tracked
            by tracker in consecutive frames.
        iou_score (float): Intersection over union score.
        data_output_format (str): Output format for data in tracker.
            Default is ``mot_challenge``.
        kwargs (dict): Additional key word arguments.
    """

    count = 0

    metadata = dict(
        data_output_formats=['mot_challenge']
    )
    # pylint: disable=too-many-arguments
    def __init__(
            self,
            track_id: int,
            frame_id: int,
            bbox: np.ndarray,
            detection_confidence: float,
            class_id: Union[str, int] = None,
            lost: int = 0,
            iou_score: float = 0.,
            data_output_format: str = 'mot_challenge',
            **kwargs: Dict[Any, Any]) -> None:
        assert data_output_format in Track.metadata['data_output_formats']
        Track.count += 1
        self.id_num = track_id

        self.detection_confidence_max = 0.
        self.lost = 0
        self.age = 0

        self.update(frame_id, bbox, detection_confidence, class_id=class_id,
                    lost=lost, iou_score=iou_score, **kwargs)

        if data_output_format == 'mot_challenge':
            self.output = self.get_mot_challenge_format
        else:
            raise NotImplementedError

    # pylint: disable=too-many-arguments
    def update(self,
               frame_id: int,
               bbox: np.ndarray,
               detection_confidence: float,
               class_id: Union[str, int] = None,
               lost: int = 0,
               iou_score: float = 0., **kwargs: Dict[Any, Any]) -> None:
        """
        Update the track.

        Args:
            frame_id (int): Camera frame id.
            bbox (numpy.ndarray): Bounding box pixel coordinates as
                (xmin, ymin, width, height) of the track.
            detection_confidence (float): Detection confidence of the object (probability).
            class_id (int or str): Class label id.
            lost (int): Number of times the object or track was not tracked
                by tracker in consecutive frames.
            iou_score (float): Intersection over union score.
            kwargs (dict): Additional key word arguments.
        """
        self.class_id = class_id
        self.bbox = np.array(bbox)
        self.detection_confidence = detection_confidence
        self.frame_id = frame_id
        self.iou_score = iou_score

        if lost == 0:
            self.lost = 0
        else:
            self.lost += lost

        for key, val in kwargs.items():
            setattr(self, key, val)

        self.detection_confidence_max = max(self.detection_confidence_max,
                                            detection_confidence)

        self.age += 1

    @property
    def centroid(self) -> np.ndarray:
        """
        Return the centroid of the bounding box.

        Returns:
            numpy.ndarray: Centroid (x, y) of bounding box.

        """
        return np.array((self.bbox[0]+0.5*self.bbox[2], self.bbox[1]+0.5*self.bbox[3]))

    def get_mot_challenge_format(self) -> \
                Tuple[int, int, Any, Any, Any, Any, float, int, int, int]:
        """
        Get the tracker data in MOT challenge format as a tuple of elements containing
        `(frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z)`

        References:
            - Website : https://motchallenge.net/

        Returns:
            tuple: Tuple of 10 elements representing `(frame, id, bb_left,
                    bb_top, bb_width, bb_height, conf, x, y, z)`.

        """
        mot_tuple = (
            self.frame_id, self.id_num, self.bbox[0], self.bbox[1], self.bbox[2],
            self.bbox[3], self.detection_confidence, -1, -1, -1
        )
        return mot_tuple

    # pylint: disable=no-self-use
    def predict(self) -> None:
        """Implement to prediction the next estimate of track."""
        # pylint: disable=notimplemented-raised
        # pylint: disable=raising-bad-type
        raise NotImplemented

    @staticmethod
    def print_all_track_output_formats() -> None:
        """Prints all metadata of track output."""
        print(Track.metadata['data_output_formats'])
