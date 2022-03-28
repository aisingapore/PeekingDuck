# Modifications copyright 2022 AI Singapore
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

"""Data class to store information of a single tracked detection."""

import numpy as np


class Track:  # pylint: disable=too-few-public-methods
    """Stores information for each tracked detection.

    Args:
        track_id (int): Tracking ID of the detection.
        bbox (np.ndarray): Bounding box coordinates with (t, l, w, h) format
            where (t, l) is the top-left corner, w is the width, and h is the
            height.

    Attributes:
        bbox (np.ndarray): Bounding box coordinates with (t, l, w, h) format
            where (t, l) is the top-left corner, w is the width, and h is the
            height.
        iou_score (float): The Intersection-over-Union value between the
            current `bbox` and the immediate previous `bbox`.
        lost (int): The number of consecutive frames where this detection is
            not found in the video frame.
        track_id (int): Tracking ID of the detection.
    """

    def __init__(self, track_id: int, bbox: np.ndarray) -> None:
        self.track_id = track_id
        self.lost = 0
        self.update(bbox)

    def update(self, bbox: np.ndarray, iou_score: float = 0.0) -> None:
        """Updates the tracking result with information from the latest frame.

        Args:
            bbox (np.ndarray): Bounding box with format (t, l, w, h) where
                (t, l) is the top-left corner, w is the width, and h is the
                height.
            iou_score (float): Intersection-over-Union between the current
                detection bounding box and its last detected bounding box.
        """
        self.bbox = bbox
        self.iou_score = iou_score
        self.lost = 0
