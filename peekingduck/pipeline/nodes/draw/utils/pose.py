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
Functions for drawing pose keypoints and connections
"""

from typing import Any, Dict, Iterable, Tuple, Union

import cv2
import numpy as np

from peekingduck.pipeline.nodes.base import ThresholdCheckerMixin
from peekingduck.pipeline.nodes.draw.utils.constants import THICK
from peekingduck.pipeline.nodes.draw.utils.general import (
    get_image_size,
    project_points_onto_original_image,
)


class Pose(ThresholdCheckerMixin):
    """Pose class to draw pose keypoints and connections"""

    def __init__(
        self,
        config: Dict[str, Any],
    ) -> None:
        self.config = config
        self.check_bounds(["keypoint_dot_color", "keypoint_connect_color"], "[0, 255]")
        self.check_bounds("keypoint_dot_radius", "[0, +inf)")

        self.keypoint_dot_color = tuple(self.config["keypoint_dot_color"])
        self.keypoint_connect_color = tuple(self.config["keypoint_connect_color"])
        self.keypoint_dot_radius = self.config["keypoint_dot_radius"]

    def draw_human_poses(
        self,
        image: np.ndarray,
        all_keypoints: np.ndarray,
        all_keypoint_connections: np.ndarray,
    ) -> None:
        # pylint: disable=too-many-arguments
        """Draw poses onto an image frame.

        Args:
            image (np.ndarray): image of current frame
            all_keypoints (np.ndarray): keypoint coordinates of shape (N, 17, 2)
                where N is the number of humans detected.
            all_keypoint_connections (np.ndarray): keypoint connections of shape
                (N, 15, 2, 2) where N is the number of humans detected.
        """
        image_size = get_image_size(image)
        num_persons = all_keypoints.shape[0]
        if num_persons > 0:
            for keypoints, keypoint_connections in zip(
                all_keypoints, all_keypoint_connections
            ):
                self._draw_connections(image, keypoint_connections, image_size)
                self._draw_keypoints(image, keypoints, image_size)

    def _draw_connections(
        self,
        image: np.ndarray,
        keypoint_connections: Union[None, Iterable[Any]],
        image_size: Tuple[int, int],
    ) -> None:
        """Draw connections between detected keypoints"""
        if keypoint_connections is not None:
            for connection in keypoint_connections:
                pt1, pt2 = project_points_onto_original_image(connection, image_size)
                cv2.line(
                    image,
                    (pt1[0], pt1[1]),
                    (pt2[0], pt2[1]),
                    self.keypoint_connect_color,
                    THICK,
                )

    def _draw_keypoints(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        image_size: Tuple[int, int],
    ) -> None:
        # pylint: disable=too-many-arguments
        """Draw detected keypoints"""
        image_keypoints = project_points_onto_original_image(keypoints, image_size)

        for _, image_keypoint in enumerate(image_keypoints):
            self._draw_one_keypoint_dot(image, image_keypoint)

    def _draw_one_keypoint_dot(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
    ) -> None:
        """Draw single keypoint"""
        cv2.circle(
            image,
            (keypoints[0], keypoints[1]),
            self.keypoint_dot_radius,
            self.keypoint_dot_color,
            -1,
        )
