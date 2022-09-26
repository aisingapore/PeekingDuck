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

from typing import List

import numpy as np


class BodyKeypoint:
    """Performs various post processing functions on the COCO format 17-keypoint
    poses. Converts keypoints to the COCO format if `keypoint_map` is provided.
    """

    # fmt: off
    SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    # fmt: on

    def __init__(self, keypoint_map: List[int] = None) -> None:
        if keypoint_map is not None and len(keypoint_map) != 17:
            raise ValueError("keypoint_map should contain 17 elements.")
        self.keypoint_map = keypoint_map

    @property
    def bboxes(self) -> np.ndarray:
        """Returns the bounding boxes encompassing each detected pose."""
        if len(self._poses) == 0:
            return np.empty((0, 4))
        return np.array(
            [
                [
                    pose[:, 0].min(),
                    pose[:, 1].min(),
                    pose[:, 0].max(),
                    pose[:, 1].max(),
                ]
                for pose in self.keypoints
            ]
        )

    @property
    def connections(self) -> np.ndarray:
        """Returns the keypoint connections."""
        return np.array(
            [
                [
                    np.vstack((keypoints[start - 1], keypoints[stop - 1]))
                    for start, stop in self.SKELETON
                ]
                for keypoints in self._poses
            ]
        )

    @property
    def keypoints(self) -> np.ndarray:
        """Returns COCO format 17 keypoints as a NumPy array."""
        return np.array(self._poses)

    def convert_scores(self, scores_list: List[List[float]]) -> np.ndarray:
        """Converts the provided keypoint scores to COCO 17-keypoint format.

        Args:
            scores_list (List[List[float]]): A (N,K,1) array of scores where N
                is the number of poses, K is the number of keypoints. K=17 for
                the COCO format.

        Returns:
            (np.ndarray): A (N, 17, 1) array of keypoint scores.
        """
        if self.keypoint_map is None:
            return np.array(scores_list)
        return np.array(
            [[scores[i] for i in self.keypoint_map] for scores in scores_list]
        )

    def update_keypoints(self, poses: List[List[List[float]]]) -> None:
        """Updates internal `_poses`. Convert to COCO format if
        `self.keypoint_map` is set.

        Args:
            poses (List[List[List[float]]]): A (N,K,2) array of keypoints where
                N is the number of poses, K is the number of keypoints. K=17 for
                the COCO format.
        """
        if self.keypoint_map is None:
            self._poses = poses
        else:
            self._poses = [[pose[i] for i in self.keypoint_map] for pose in poses]
