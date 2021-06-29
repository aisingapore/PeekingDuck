# Copyright 2021 AI Singapore
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

"""
Estimates the 3D coordinates of a human given 2D pose coordinates
"""

from typing import Dict, Any
import numpy as np
from peekingduck.pipeline.nodes.node import AbstractNode

NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_PELVIS = 11
RIGHT_PELVIS = 12
TORSO_KEYPOINTS = [
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_PELVIS, RIGHT_PELVIS]


class Node(AbstractNode):
    """Node that uses pose keypoint information of torso to estimate 3d location"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)

        self.torso_factor = config['torso_factor']
        self.focal_length = config['focal_length']

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Converts pose keypoints into 3D locations.

        Args:
            inputs (dict): Dict with keys "keypoints".

        Returns:
            outputs (dict): Dict with keys "obj_3D_locs".
        """

        locations = []

        for keypoints in inputs["keypoints"]:
            torso_keypoints = self._get_torso_keypoints(keypoints)
            if self._enough_torso_keypoints(torso_keypoints):
                bbox = self._get_bbox(torso_keypoints)
            else:
                bbox = self._get_bbox(keypoints)

            point = self._get_3d_point_from_bbox(
                bbox, self.focal_length, self.torso_factor)
            locations.append(point)

        outputs = {"obj_3D_locs": locations}

        return outputs

    @staticmethod
    def _get_torso_keypoints(keypoints: np.array) -> np.array:
        """Filter keypoints to get only selected keypoints for torso"""

        torso_keypoints = keypoints[TORSO_KEYPOINTS, :]  # type: ignore
        # ignore keypoints that are '-1.' as below confidence score and are masked
        torso_keypoints = np.reshape(
            torso_keypoints[torso_keypoints != -1.], (-1, 2))

        return torso_keypoints

    @staticmethod
    def _enough_torso_keypoints(torso_keypoints: np.array) -> bool:
        """Returns False if not enough keypoints to represent torso"""

        if torso_keypoints.shape[0] >= 2:
            return True
        return False

    @staticmethod
    def _get_bbox(keypoints: np.array) -> np.array:
        """Get coordinates of a bbox around keypoints"""

        top_left_x, top_left_y = keypoints.min(axis=0)
        btm_right_x, btm_right_y = keypoints.max(axis=0)

        return np.array([top_left_x, top_left_y, btm_right_x, btm_right_y])

    @staticmethod
    def _get_3d_point_from_bbox(bbox: np.array, focal_length: float,
                                torso_factor: float) -> np.array:
        """Get the 3d coordinates of the centre of a bounding box"""

        # Subtraction is to make the camera the origin of the coordinate system
        center_2d = ((bbox[0:2] + bbox[2:4]) * 0.5) - np.array([0.5, 0.5])
        torso_height = bbox[3] - bbox[1]

        z_coord = (focal_length * torso_factor) / torso_height
        x_coord = (center_2d[0] * torso_factor) / torso_height
        y_coord = (center_2d[1] * torso_factor) / torso_height

        return np.array([x_coord, y_coord, z_coord])
