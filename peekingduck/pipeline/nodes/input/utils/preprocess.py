"""
Copyright 2021 AI Singapore

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from typing import Any, Tuple
import numpy as np
import cv2

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def set_res(stream: Any, desired_width: int, desired_height: int) -> None:
    '''
    Sets the resolution for the video frame
    '''
    stream.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    actual_width, actual_height = get_res(stream)
    if desired_width != actual_width:
        logger.warning("Unable to change width of video frame to %s, current width: %s!",
                       desired_width, actual_width)
    if desired_height != actual_height:
        logger.warning("Unable to change height of video frame to %s, current height: %s!",
                       desired_height, actual_height)


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
