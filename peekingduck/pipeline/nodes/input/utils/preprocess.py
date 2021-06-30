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
Preprocessing fuctions for input nodes
"""

import logging
from typing import Any, Tuple
import numpy as np
import cv2

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_res(stream: Any) -> Tuple[int, int]:
    '''
    Gets the resolution for the video frame
    '''
    width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return width, height


def mirror(frame: np.array) -> np.array:
    '''
    Mirrors a video frame.
    '''
    return cv2.flip(frame, 1)


def resize_image(frame: np.array, desired_width: int, desired_height: int) -> Any:
    """function that resizes the image input
    to the desired dimensions

    Args:
        frame (np.array): image
        desired_width: width of the resized image
        desired_height: height of the resized image

    Returns:
        image (np.array): returns a scaled image depending on the
        desired wight and height
    """
    return cv2.resize(frame, (desired_width, desired_height))
