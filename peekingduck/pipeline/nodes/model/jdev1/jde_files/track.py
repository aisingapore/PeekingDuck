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

"""Track states and STrack to store information for each tracked detection.

Modifications include:
- Make TrackState inherit Enum for clarity
- Make BaseTrack an ABC and use abstractmethod decorator for clarity
    - Removed update() method as it's not used in the current implementation of
        JDE.
- Renamed tlbr to xyxy for consistency with other model nodes.
- Removed new_id argument from re_activate() since it's never used
"""

from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from typing import Deque, List

import numpy as np
import torch

from peekingduck.pipeline.nodes.model.jdev1.jde_files.kalman_filter import KalmanFilter


class TrackState(Enum):
    """Numbered states of Track.

    Attributes:
        NEW: The Track is newly created.
        TRACKED: The Track is actively tracked.
        LOST: The Track is not found among the detections and is considered
            "lost".
        REMOVED: The Track has been lost for longer than the threshold and is
            to be removed.
    """

    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class BaseTrack(ABC):
    """Base Tracking class."""

    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.NEW

    features: Deque[np.ndarray] = deque([])
    curr_feature = None
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    @property
    def end_frame(self) -> int:
        """The last frame ID where this is actively tracked."""
        return self.frame_id

    def mark_lost(self) -> None:
        """Marks the Track as lost."""
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        """Marks the Track for removal."""
        self.state = TrackState.REMOVED

    @abstractmethod
    def activate(self, kalman_filter: KalmanFilter, frame_id: int) -> None:
        """Starts a new tracklet.

        Args:
            kalman_filter (KalmanFilter): Kalman filter for state estimation.
            frame_id (int): Current frame ID.
        """

    @abstractmethod
    def update(
        self, new_track: "STrack", frame_id: int, update_feature: bool = True
    ) -> None:
        """Updates a matched track.

        Args:
            new_track (STrack): New STrack.
            frame_id (int): Frame ID.
            update_feature (bool, optional): Update feature. Defaults to True.
        """

    @staticmethod
    def next_id() -> int:
        """The next track ID."""
        BaseTrack._count += 1
        return BaseTrack._count


class STrack(BaseTrack):  # pylint: disable=too-many-instance-attributes
    """Handles information of a single track.

    Args:
        tlwh (np.ndarray): Bounding box in (top left x, top left y, width,
            height) format.
        score (torch.Tensor): Detection confidence score.
        feat (np.ndarray): Embeddings.
        buffer_size (int): Maximum of past embeddings to store.
    """

    def __init__(
        self,
        tlwh: np.ndarray,
        score: torch.Tensor,
        feat: np.ndarray,
        buffer_size: int = 30,
    ) -> None:
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.score = score

        self.kalman_filter: KalmanFilter
        self.mean = None
        self.covariance = None

        self.is_activated = False
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def __repr__(self) -> str:
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"

    @property
    def tlwh(self) -> np.ndarray:
        """The current position in bounding box format `(top left x,
        top left y, width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyah(self) -> np.ndarray:
        """The current position in bounding box to format `(center x, center y,
        aspect ratio, height)`, where the aspect ratio is `width / height`.
        """
        return self.tlwh2xyah(self.tlwh)

    @property
    def xyxy(self) -> np.ndarray:
        """The current position in bounding box format `(x1, y1, x2, y2)` where
        (x1, y1) is top left and (x2, y2) is bottom right.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def activate(self, kalman_filter: KalmanFilter, frame_id: int) -> None:
        """Starts a new tracklet.

        Args:
            kalman_filter (KalmanFilter): Kalman filter for state estimation.
            frame_id (int): Current frame ID.
        """
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh2xyah(self._tlwh)
        )

        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        self.frame_id = frame_id
        self.start_frame = frame_id

    def update(
        self, new_track: "STrack", frame_id: int, update_feature: bool = True
    ) -> None:
        """Updates a matched track.

        Args:
            new_track (STrack): New STrack.
            frame_id (int): Frame ID.
            update_feature (bool, optional): Update feature. Defaults to True.
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh2xyah(new_tlwh)
        )
        self.state = TrackState.TRACKED
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    def re_activate(self, new_track: "STrack", frame_id: int) -> None:
        """Re-activates STrack.

        Args:
            new_track (STrack): New STrack.
            frame_id (int): Current frame ID.
        """
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh2xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id

    def update_features(self, feat: np.ndarray) -> None:
        """Updates the features (embeddings).

        Args:
            feat (np.ndarray): Embeddings.
        """
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    @staticmethod
    def multi_predict(stracks: List["STrack"], kalman_filter: KalmanFilter) -> None:
        """Runs the vectorised version of Kalman filter prediction step.

        Args:
            stracks (List[STrack]): List of STrack.
            kalman_filter (KalmanFilter): Kalman filter for state estimation.
        """
        if not stracks:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])  # type: ignore
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, strack in enumerate(stracks):
            if strack.state != TrackState.TRACKED:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = kalman_filter.multi_predict(
            multi_mean, multi_covariance
        )
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def tlwh2xyah(tlwh: np.ndarray) -> np.ndarray:
        """Converts bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.

        Args:
            tlwh (np.ndarray): Input bounding box with format `(top left x,
                top left y, width, height)`.

        Returns:
            (np.ndarray): Bounding box with (x, y, a, h) format.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def xyxy2tlwh(xyxy: np.ndarray) -> np.ndarray:
        """Converts bounding box to format `(top left x, top left y, width,
        height)`.

        Args:
            xyxy (np.ndarray): Input bounding box with format (x1, y1, x2, y2)
                where (x1, y1) is top left, (x2, y2) is bottom right.

        Returns:
            (np.ndarray): Bounding box with `(top left x, top left y, width,
                height)` format.
        """
        ret = np.asarray(xyxy).copy()
        ret[2:] -= ret[:2]
        return ret
