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
Creats a zone from a polygon area
"""

from typing import List, Tuple
from shapely.geometry.polygon import Polygon, Point


class Zone:
    """This class uses polygon area to create a zone for counting.
    """

    def __init__(self, coord_list: List[List[float]]) -> None:
        # Each zone is a polygon created by a list of x, y coordinates
        self.polygon_points = [tuple(x) for x in coord_list]
        self.polygon = Polygon(self.polygon_points)

    def point_within_zone(self, x_coord: float, y_coord: float) -> bool:
        """Function used to check whether the bottom middle point of the bounding box
        is within the stipulated zone created by the divider.

        Args:
            x (float): middle x position of the bounding box
            y (float): lowest y position of the bounding box

        Returns:
            boolean: whether the point given is within the zone.
        """
        return self._is_inside(x_coord, y_coord)

    def get_all_points_of_area(self) -> List[Tuple[float, ...]]:
        """Function used to Get all (x, y) tuple points that form the area of the zone.

        Args:
            None

        Returns:
            list: returns a list of (x, y) points that form the zone area.
        """
        return self.polygon_points

    def _is_inside(self, x_coord: float, y_coord: float) -> bool:
        point = Point((x_coord, y_coord))
        return self.polygon.buffer(1).contains(point)
