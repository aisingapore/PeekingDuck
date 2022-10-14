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
Pose class for drawing pose keypoints and connections.
"""

from typing import Any, Iterable, Optional, Tuple

import cv2
import numpy as np

from peekingduck.nodes.draw.utils.constants import THICK
from peekingduck.nodes.draw.utils.general import (
    get_image_size,
    project_points_onto_original_image,
)


class Pose:  # pylint: disable=too-few-public-methods
    """Pose class to draw pose keypoints and connections."""

    def __init__(
        self,
        keypoint_dot_color: Tuple[int, int, int],
        keypoint_connect_color: Tuple[int, int, int],
        keypoint_dot_radius: int,
    ) -> None:
        self.keypoint_dot_color = keypoint_dot_color
        self.keypoint_connect_color = keypoint_connect_color
        self.keypoint_dot_radius = keypoint_dot_radius

    def draw_human_poses(
        self,
        image: np.ndarray,
        all_keypoints: np.ndarray,
        all_keypoint_connections: np.ndarray,
    ) -> None:
        """Draw poses onto an image frame.

        Args:
            image (np.ndarray): Input image frame.
            all_keypoints (np.ndarray): Keypoint coordinates of shape (N, 17, 2)
                where N is the number of humans detected.
            all_keypoint_connections (np.ndarray): Keypoint connections of shape
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
        keypoint_connections: Optional[Iterable[Any]],
        image_size: Tuple[int, int],
    ) -> None:
        """Draw connections between detected keypoints.

        Args:
            image (np.ndarray): Input image frame.
            keypoint_connections (Optional[Iterable[Any]]): Keypoint connections
                of one individual human with shape (15, 2, 2).
            image_size (Tuple[int, int]): Image size of the input image frame.
        """
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
        """Draw detected keypoints.

        Args:
            image (np.ndarray): Input image frame.
            keypoints (np.ndarray): Keypoint coordinates of one individual human
                with shape (17, 2).
            image_size (Tuple[int, int]): Image size of the input image frame.
        """
        image_keypoints = project_points_onto_original_image(keypoints, image_size)

        for _, image_keypoint in enumerate(image_keypoints):
            self._draw_one_keypoint_dot(image, image_keypoint)

    def _draw_one_keypoint_dot(
        self,
        image: np.ndarray,
        keypoint: np.ndarray,
    ) -> None:
        """Draw single keypoint.

        Args:
            image (np.ndarray): Input image frame.
            keypoint (np.ndarray): Keypoint coordinates of shape (2,).
        """
        cv2.circle(
            image,
            (keypoint[0], keypoint[1]),
            self.keypoint_dot_radius,
            self.keypoint_dot_color,
            -1,
        )
