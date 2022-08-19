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

"""
Estimates the 3D coordinates of a person given 2D pose coordinates.
"""

from typing import Any, Dict

import numpy as np

from peekingduck.pipeline.nodes.abstract_node import AbstractNode

NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_PELVIS = 11
RIGHT_PELVIS = 12
TORSO_KEYPOINTS = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_PELVIS, RIGHT_PELVIS]


class Node(AbstractNode):
    """Uses pose keypoint information of the torso to estimate 3D location.

    Inputs:
        |keypoints_data|

    Outputs:
        |obj_3D_locs_data|

    Configs:
        focal_length (:obj:`float`): **default = 1.14**. |br|
            Approximate focal length of webcam used, in metres. Example on
            measuring focal length can be found `here <https://learnopencv.com
            /approximate-focal-length-for-webcams-and-cell-phone-cameras/>`_.
        torso_factor (:obj:`float`): **default = 0.9**. |br|
            A factor used to estimate real-world distance from pixels, based on
            average human torso length in metres. The value varies across
            different camera set-ups, and calibration may be required.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Converts pose keypoints into 3D locations."""
        locations = []

        for keypoints in inputs["keypoints"]:
            torso_keypoints = self._get_torso_keypoints(keypoints)
            if self._enough_torso_keypoints(torso_keypoints):
                bbox = self._get_bbox(torso_keypoints)
            else:
                bbox = self._get_bbox(keypoints)

            point = self._get_3d_point_from_bbox(
                bbox, self.focal_length, self.torso_factor
            )
            locations.append(point)

        outputs = {"obj_3D_locs": locations}

        return outputs

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {"focal_length": float, "torso_factor": float}

    @staticmethod
    def _get_torso_keypoints(keypoints: np.ndarray) -> np.ndarray:
        """Filter keypoints to get only selected keypoints for torso"""
        torso_keypoints = keypoints[TORSO_KEYPOINTS, :]  # type: ignore
        # ignore keypoints that are '-1.' as below confidence score and are masked
        torso_keypoints = np.reshape(torso_keypoints[torso_keypoints != -1.0], (-1, 2))

        return torso_keypoints

    @staticmethod
    def _enough_torso_keypoints(torso_keypoints: np.ndarray) -> bool:
        """Returns False if not enough keypoints to represent torso"""
        if torso_keypoints.shape[0] >= 2:
            return True
        return False

    @staticmethod
    def _get_bbox(keypoints: np.ndarray) -> np.ndarray:
        """Get coordinates of a bbox around keypoints"""
        top_left_x, top_left_y = keypoints.min(axis=0)
        btm_right_x, btm_right_y = keypoints.max(axis=0)

        return np.array([top_left_x, top_left_y, btm_right_x, btm_right_y])

    @staticmethod
    def _get_3d_point_from_bbox(
        bbox: np.ndarray, focal_length: float, torso_factor: float
    ) -> np.ndarray:
        """Get the 3d coordinates of the centre of a bounding box"""
        # Subtraction is to make the camera the origin of the coordinate system
        center_2d = ((bbox[0:2] + bbox[2:4]) * 0.5) - np.array([0.5, 0.5])
        torso_height = bbox[3] - bbox[1]

        z_coord = (focal_length * torso_factor) / torso_height
        x_coord = (center_2d[0] * torso_factor) / torso_height
        y_coord = (center_2d[1] * torso_factor) / torso_height

        return np.array([x_coord, y_coord, z_coord])
