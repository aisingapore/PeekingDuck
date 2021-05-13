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
from __future__ import annotations
from typing import List, Any, Tuple
from peekingduck.pipeline.nodes.heuristic.zoningv1.zone import Zone


class DividerZone(Zone):
    """This is a zone subclass that uses dividers to create a zone.
    """
    def __init__(self, coord_conditions: List[Any]) -> None:
        super().__init__("divider")

        # each coord pairs is an array of x1, y1, x2, y2, condition
        self.dividers = [Divider(x_1, y_1, x_2, y_2, condition) for \
            x_1, y_1, x_2, y_2, condition in coord_conditions]

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

    def get_dividers(self) -> List[Divider]:
        """Getter of Dividers used in defining the zone.

        Returns:
            dividers (Divider): returns dividers used to define the zone.
        """
        return self.dividers

    def _is_inside(self, x_coord: float, y_coord: float) -> bool:
        """ Funciton looks at whether the given xy point is within the
        specified zone given the dividers setup for the zone
        """
        for divider in self.dividers:
            if not divider.point_match_divider_condition(x_coord, y_coord):
                return False
        # if pass for loop, point passes all conditions, means is within dividers
        return True


class Divider:
    """A class to do rule base check for people counting with dividers set in config
    """

    # pylint: disable=too-many-arguments
    def __init__(self, cord1_x1: float, cord_y1: float,
                 cord_x2: float, cord_y2: float, inside_cond: str) -> None:
        self.inside_cond = inside_cond
        self.point_1 = (cord1_x1, cord_y1)
        self.point_2 = (cord_x2, cord_y2)

        if self.point_1[0] != self.point_2[0]:
            if self.point_1[1] != self.point_2[1]:
                # Will need to use gradient and y_intercept to check
                self.gradient = (cord_y2 - cord_y1)/(cord_x2 - cord1_x1)
                self.y_intercept = cord_y1 - (cord1_x1 * self.gradient)

    def point_match_divider_condition(self, x_coord: float, y_coord: float) -> bool:
        """Function used to check whether the bottom middle point of the bounding box
        is within the stipulated zone created by the divider.

        Args:
            x (float): middle x position of the bounding box
            y (float): lowest y position of the bounding box

        Returns:
            boolean: whether the point given is within the zone.
        """
        return self._is_inside(x_coord, y_coord)

    def get_end_points_to_draw_on_frame(self, max_x: float, max_y: float) \
        -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Uses the maximum x and y given to return two points
        which would draw the divider onto the frame

        Args:
            max_x (float): full width of the frame
            max_y (float): full height of the frame

        Returns:
            tuples: returns two tuple of the two plot points in (x, y) form
        """

        if self.point_1[0] == self.point_2[0]:
            return (int(self.point_1[0]), 0), (int(self.point_1[0]), int(max_y))
        if self.point_1[1] == self.point_2[1]:
            return (0, int(self.point_1[1])), (int(max_x), int(self.point_1[1]))

        draw_points = []
        # check the two points which the line intercepts the borders of the frame and use
        # it to draw the diagonal line zone boundary
        if 0 <= self.y_intercept <= max_y:
            draw_points.append((0, int(self.y_intercept)))
        x_intercept = -(self.y_intercept) / self.gradient
        if 0 <= x_intercept <= max_x:
            draw_points.append((int(x_intercept), 0))
        y_max_intercept = max_x * self.gradient + self.y_intercept
        if 0 <= y_max_intercept <= max_y:
            draw_points.append((int(max_x), int(y_max_intercept)))
        x_max_intercept = (max_y - self.y_intercept) / self.gradient
        if 0 <= x_max_intercept <= max_x:
            draw_points.append((int(x_max_intercept), int(max_y)))

        return draw_points[0], draw_points[1]

    def _is_inside(self, x_coord: float, y_coord: float) -> bool:
        # condition when line is horizontal
        if self.point_1[0] == self.point_2[0]:
            if self.inside_cond == "greater":
                return x_coord >= self.point_1[0]
            if self.inside_cond == "smaller":
                return x_coord <= self.point_1[0]
            raise TypeError("line condition for divider should be greater or smaller.")

        # condition when line is vertical
        if self.point_1[1] == self.point_2[1]:
            if self.inside_cond == "greater":
                return y_coord >= self.point_1[1]
            if self.inside_cond == "smaller":
                return y_coord <= self.point_1[1]
            raise TypeError("line condition for divider should be greater or smaller.")

        #condition when line is diagonal
        value_to_check = (self.gradient * x_coord) + self.y_intercept
        if self.inside_cond == "greater":
            return y_coord >= value_to_check
        if self.inside_cond == "smaller":
            return y_coord <= value_to_check
        raise TypeError("line condition for divider should be greater or smaller.")
