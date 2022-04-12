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
Custom PNG reader to fix opencv 'PNG magic' problem on Windows platform
"""
from typing import Any, Tuple
import cv2
import numpy as np


class PNGReader:
    """Custom PNG reader to fix opencv 'PNG magic' problem on Windows platform"""

    def __init__(self, input_source: str) -> None:
        self.img = cv2.imread(input_source)
        self.height, self.width, _ = self.img.shape
        self.get_map = {
            cv2.CAP_PROP_FPS: 0,
            cv2.CAP_PROP_FRAME_COUNT: 1,
            cv2.CAP_PROP_FRAME_WIDTH: self.width,
            cv2.CAP_PROP_FRAME_HEIGHT: self.height,
        }
        self.has_frames: bool = True

    def get(self, param: Any) -> int:
        """To mimic opencv's video capture object get(cv2.SOME_PROPERTY)

        Args:
            param (Any): cv2 property

        Returns:
            int: value of cv2 property if supported, otherwise -1
        """
        if param in self.get_map:
            return self.get_map[param]
        return -1

    def isOpened(self) -> bool:  # pylint: disable=invalid-name, no-self-use
        """To mimic opencv's video capture object isOpened()

        Returns:
            bool: always True
        """
        return True

    def read(self) -> Tuple[bool, np.ndarray]:
        """To mimic opencv's video capture object read()

        Returns:
            Tuple[bool, np.ndarray]: tuple of return status, image frame
        """
        if self.has_frames:
            self.has_frames = False  # only 1 frame to read and return
            return True, self.img
        return False, None

    def release(self) -> None:
        """To mimic opencv's video capture object release().
        Dummy method does nothing.
        """
