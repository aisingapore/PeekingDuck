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
Crop bounding box from image.
"""

from typing import Tuple
import numpy as np


def crop_bbox(
    frame: np.array, bbox: np.array, original_h: int, original_w: int
) -> Tuple[int, int, int]:
    """Function that crops a bounding box from an image.

    Args:
        frame (np.array): Image from video frame.
        bbox (np.array): Individual detected bounding box.
        original_h (int): Height of frame.
        original_w (int): Width of frame.

    Returns:
        Tuple[int, int, int]: Cropped bounding box from image frame.
    """
    ystart = bbox[1] * original_h
    ystop = bbox[3] * original_h
    xstart = bbox[0] * original_w
    xstop = bbox[2] * original_w

    # Using numpy slicing
    crop = frame[int(ystart) : int(ystop), int(xstart) : int(xstop)]

    return crop
