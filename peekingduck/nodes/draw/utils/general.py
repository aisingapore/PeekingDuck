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
General utils for drawing functions
"""

from typing import List, Tuple, Union

import numpy as np

from peekingduck.nodes.draw.utils.constants import COLOR_MAP


def get_image_size(frame: np.ndarray) -> Tuple[int, int]:
    """Obtain image size of input frame

    Args:
        frame (np.array): image of current frame

    Returns:
        image_size (Tuple[int, int]): Width and height of image
    """
    image_size = (frame.shape[1], frame.shape[0])  # width, height
    return image_size


def project_points_onto_original_image(
    points: np.ndarray, image_size: Tuple[int, int]
) -> np.ndarray:
    """Project points from relative value (0, 1) to absolute values in original
    image. Note that coordinate (0, 0) starts from image top-left.

    Args:
        points (np.array): points on an image
        image_size (Tuple[int, int]): Width and height of image

    Returns:
        project_points (np.ndarray): projected points on the original image
                                     in integer coordinates
    """
    if len(points) == 0:
        return []

    points = points.reshape((-1, 2))

    projected_points = np.array(points, dtype=np.float32)

    width, height = image_size[0], image_size[1]
    projected_points[:, 0] *= width
    projected_points[:, 1] *= height

    return np.round(projected_points).astype(int)


def get_color(color: Union[str, List[int]]) -> Tuple[int, int, int]:
    """Convert color to BGR tuple.

    Example:
        >>> color = "red"
        >>> get_color(color)
        (0, 0, 255)

    Args:
        color (Union[str, List[int]]): Color can be a string from COLOR_MAP
            or a list of 3 integers representing BGR values.

    Returns:
        color_tuple (Tuple[int, int, int]): Color in BGR tuple.
    """
    if isinstance(color, str):
        color_tuple = COLOR_MAP[color]
    else:
        color_tuple = (color[0], color[1], color[2])
    return color_tuple
