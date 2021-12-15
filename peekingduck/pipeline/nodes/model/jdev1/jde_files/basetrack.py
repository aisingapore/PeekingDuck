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
# Original copyright (c) 2019 ZhongdaoWang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
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
Creates BaseTrack.
"""

from typing import Any, List
import numpy as np


# pylint: disable=too-few-public-methods
class TrackState:
    """Numbered States of Track."""

    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    """Base Tracking class."""

    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    features: List[np.ndarray] = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self) -> int:
        """Used to calculate track lost state.

        Returns:
            int: Frame ID at end.
        """
        return self.frame_id

    @staticmethod
    def next_id() -> int:
        """Returns ID for next track.

        Returns:
            int: Next track ID.
        """
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args: Any) -> None:
        """Start a new track state.

        Raises:
            NotImplementedError: Raises an exception.
        """
        raise NotImplementedError

    def predict(self) -> None:
        """Predict track state.

        Raises:
            NotImplementedError: Raises an exception.
        """
        raise NotImplementedError

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update a track state.

        Raises:
            NotImplementedError: Raises an exception.
        """
        raise NotImplementedError

    def mark_lost(self) -> None:
        """Mark track as lost."""
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        """Mark track to be removed."""
        self.state = TrackState.Removed
