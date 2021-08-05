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
from cv2 import FONT_HERSHEY_SIMPLEX, LINE_AA
from peekingduck.pipeline.nodes.draw.utils.constants import \
    BLACK, WHITE, FILLED, PRIMARY_PALETTE, PRIMARY_PALETTE_LENGTH, \
        SMALL_FONTSCALE, THICK, VERY_THICK
from peekingduck.pipeline.nodes.draw.utils.general import \
    get_image_size


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


def draw_zone_count(frame: np.array, zone_count: List[int]) -> None:
    """draw pts of selected object onto frame

    Args:
        frame (np.array): image of current frame
        zone_count (List[float]): object count, likely people, of each zone used
        in the zone analytics
    """
    y_pos = 125
    _, height = get_image_size(frame)

    # Temporary code for creating legend box
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, height - y_pos), (200, height - 25), BLACK, FILLED)
    # apply the overlay
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    y_pos -= 25
    text = '--ZONE COUNTS--'
    cv2.putText(frame, text, (25, height - y_pos), FONT_HERSHEY_SIMPLEX,
                SMALL_FONTSCALE, WHITE, THICK, LINE_AA)
    for i, count in enumerate(zone_count):
        y_pos -= 25
        cv2.rectangle(frame,
                      (20, height - y_pos - 14),
                      (40, height - y_pos + 5),
                      PRIMARY_PALETTE[(i+1) % PRIMARY_PALETTE_LENGTH],
                      FILLED)
        text = ' ZONE-{0}: {1}'.format(i+1, count)
        cv2.putText(frame,
                    text,
                    (40, height - y_pos),
                    FONT_HERSHEY_SIMPLEX,
                    SMALL_FONTSCALE,
                    WHITE,
                    THICK,
                    LINE_AA)
