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
Draws utilities regarding zoning functions
"""

from typing import List, Tuple, Any
import numpy as np
import cv2
from peekingduck.pipeline.nodes.draw.utils.constants import \
    PRIMARY_PALETTE, PRIMARY_PALETTE_LENGTH, VERY_THICK


def _draw_zone_area(frame: np.array, points: List[Tuple[int]],
                    zone_index: int) -> None:
    total_points = len(points)
    for i in range(total_points):
        if i == total_points-1:
            # for last point, link to first point
            cv2.line(frame, points[i], points[0],
                     PRIMARY_PALETTE[(zone_index+1) % PRIMARY_PALETTE_LENGTH], VERY_THICK)
        else:
            # for all other points, link to next point in polygon
            cv2.line(frame, points[i], points[i+1],
                     PRIMARY_PALETTE[(zone_index+1) % PRIMARY_PALETTE_LENGTH], VERY_THICK)


def draw_zones(frame: np.array, zones: List[Any]) -> None:
    """draw the boundaries of the zones used in zoning analytics

    Args:
        frame (np.array): image of current frame
        zones (Zone): zones used in the zoning analytics. possible
        classes are Area and Divider.
    """
    for i, zone_pts in enumerate(zones):
        _draw_zone_area(frame, zone_pts, i)
