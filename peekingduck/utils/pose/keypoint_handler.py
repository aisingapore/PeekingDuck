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

"""Utility class which creates keypoint connections and bounding boxes from
pose estimation landmarks.
"""

from abc import ABC
from typing import List, Optional

import numpy as np

from peekingduck.utils.abstract_class_attributes import abstract_class_attributes


@abstract_class_attributes("NUM_KEYPOINTS", "SKELETON")
class KeypointHandler(ABC):
    """Performs various post processing functions on generic K-keypoint
    poses. Converts keypoints to another format if `keypoint_map` is provided.
    """

    def __init__(
        self,
        keypoint_map: Optional[List[int]] = None,
        score_threshold: float = 0.0,
    ) -> None:
        if keypoint_map is not None and len(keypoint_map) != self.NUM_KEYPOINTS:
            raise ValueError(
                f"keypoint_map should contain {self.NUM_KEYPOINTS} elements."
            )
        self.keypoint_map = keypoint_map
        self.score_threshold = score_threshold

    @property
    def bboxes(self) -> np.ndarray:
        """Bounding boxes encompassing each detected pose."""
        if len(self.keypoints) == 0:
            return np.empty((0, 4))

        # apply absolute when calculating min because invalid keypoints are
        # hardcoded to (-1, -1)
        return np.array(
            [
                np.hstack((np.abs(pose).min(axis=0), pose.max(axis=0)))
                for pose in self.keypoints
            ]
        )

    @property
    def connections(self) -> np.ndarray:
        """Keypoint connections, a connection is present only when both
        keypoints are valid.
        """
        return np.array(
            [
                [pose[edge] for edge in self.SKELETON if mask[edge].all()]
                for pose, mask in zip(self.keypoints, self.keypoint_masks)
            ]
        )

    @property
    def keypoints(self) -> np.ndarray:
        """Keypoints as a NumPy array."""
        return getattr(self, "_poses", np.empty(0))

    @property
    def keypoint_masks(self) -> np.ndarray:
        """Boolean mask of keypoints above the confidence threshold."""
        return getattr(self, "_pose_masks", np.empty(0))

    @property
    def scores(self) -> np.ndarray:
        """Keypoint scores as a NumPy array."""
        return getattr(self, "_pose_scores", np.empty(0))

    def update(
        self, poses: List[List[List[float]]], pose_scores: List[List[float]]
    ) -> None:
        """Updates internal `_poses`. Convert to another format if
        `self.keypoint_map` is set.

        Args:
            poses (List[List[List[float]]]): A (N,K,2) array of keypoints where
                N is the number of poses, K is the number of keypoints. K=17 for
                the COCO format.
            pose_scores (List[List[float]]): A (N,K,1) array of scores where N
                is the number of poses, K is the number of keypoints. K=17 for
                the COCO format.
        """
        self._poses = np.asarray(poses)
        self._pose_scores = np.asarray(pose_scores)
        self._pose_masks = self._pose_scores > self.score_threshold

        if self.keypoint_map is not None:
            self._poses = self._poses[:, self.keypoint_map]
            self._pose_scores = self._pose_scores[:, self.keypoint_map]
            self._pose_masks = self._pose_masks[:, self.keypoint_map]

        # Constrain keypoint coordinates to [0, 1]
        self._poses = np.clip(self._poses, 0, 1)
        # Assign low confidence keypoint coordinates as (-1, -1)
        self._poses[~self._pose_masks] = [-1, -1]


class BlazePoseBody(KeypointHandler):
    """Body keypoints in BlazePose format (33 keypoints)."""

    NUM_KEYPOINTS = 33
    # fmt: off
    SKELETON = [[0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6],
                [6, 8], [9, 10], [11, 12], [11, 13], [13, 15], [15, 17],
                [15, 19], [15, 21], [17, 19], [12, 14], [14, 16], [16, 18],
                [16, 20], [16, 22], [18, 20], [11, 23], [12, 24], [23, 24],
                [23, 25], [24, 26], [25, 27], [26, 28], [27, 29], [28, 30],
                [29, 31], [30, 32], [27, 31], [28, 32]]
    # fmt: on


class COCOBody(KeypointHandler):
    """Body keypoints in COCO format (17 keypoints)."""

    NUM_KEYPOINTS = 17
    # fmt: off
    SKELETON = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11],
                [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2],
                [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
    # fmt: on


class COCOHand(KeypointHandler):
    """Hand keypoints in COCO format (21 keypoints)."""

    NUM_KEYPOINTS = 21
    # fmt: off
    SKELETON = [[0, 1], [0, 5], [9, 13], [13, 17], [5, 9], [0, 17],
                [1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [9, 10],
                [10, 11], [11, 12], [13, 14], [14, 15], [15, 16], [17, 18],
                [18, 19], [19, 20]]
    # fmt: on
