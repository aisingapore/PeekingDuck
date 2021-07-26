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
functions for drawing legend related UI components
"""

from typing import Dict, List, Any
import numpy as np
import cv2
from cv2 import FONT_HERSHEY_SIMPLEX, LINE_AA
from peekingduck.pipeline.nodes.draw.utils.constants import \
    THICK, WHITE, SMALL_FONTSCALE, BLACK, FILLED, \
    PRIMARY_PALETTE, PRIMARY_PALETTE_LENGTH
from peekingduck.pipeline.nodes.draw.utils.general import get_image_size


class Legend:
    """Legend class that uses available info to draw legend box on frame"""

    def __init__(self) -> None:
        self.func_reg = self._get_legend_registry()
        self.legend_left_x = 15

        self.frame = None
        self.legend_starting_y = 0
        self.delta_y = 0
        self.legend_height = 0

    def draw(self, inputs: Dict[str, Any], items: List[str], position: str) -> np.array:
        """ Draw legends onto image

        Args:
            inputs (dict): dictionary of all available inputs for drawing the legend
            items (list): list of items to be drawn in legend
            position (str): used to control whether legend box is drawn on top or bottom
        """
        self.frame = inputs['img']

        self.legend_height = self._get_legend_height(inputs, items)
        self._set_legend_variables(position)

        self._draw_legend_box(self.frame)
        y_pos = self.legend_starting_y + 20
        for item in items:
            self.func_reg[item](self.frame, y_pos, inputs[item])
            y_pos += 20

    def _draw_count(self, frame: np.array, y_pos: int, count: int) -> None:
        """draw count of selected object onto frame

        Args:
            frame (np.array): image of current frame
            y_pos (int): y_position to draw the count text
            count (int): total count of selected object
                in current frame
        """
        text = 'COUNT: {0}'.format(count)
        cv2.putText(frame, text, (self.legend_left_x + 10, y_pos), FONT_HERSHEY_SIMPLEX,
                    SMALL_FONTSCALE, WHITE, THICK, LINE_AA)

    def _draw_fps(self, frame: np.array, y_pos: int, current_fps: float) -> None:
        """ Draw FPS onto frame image

        Args:
            frame (np.array): image of current frame
            y_pos (int): y position to draw the count info text
            current_fps (float): value of the calculated FPS
        """
        text = "FPS: {:.05}".format(current_fps)

        cv2.putText(frame, text, (self.legend_left_x + 10, y_pos), FONT_HERSHEY_SIMPLEX,
                    SMALL_FONTSCALE, WHITE, THICK, LINE_AA)

    def _draw_zone_count(self, frame:np.array, y_pos: int, counts: List[int]) -> None:
        """ Draw zone counts of all zones onto frame image

        Args:
            frame (np.array): image of current frame
            y_pos (int): y position to draw the count info text
            counts (list): list of zone counts
        """
        text = '-ZONE COUNTS-'
        cv2.putText(frame, text, (self.legend_left_x + 10, y_pos), FONT_HERSHEY_SIMPLEX,
                    SMALL_FONTSCALE, WHITE, THICK, LINE_AA)
        for i, count in enumerate(counts):
            y_pos += 20
            cv2.rectangle(frame,
                        (self.legend_left_x + 10, y_pos - 14),
                        (self.legend_left_x + 30, y_pos + 5),
                        PRIMARY_PALETTE[(i+1) % PRIMARY_PALETTE_LENGTH],
                        FILLED)
            text = ' ZONE-{0}: {1}'.format(i+1, count)
            cv2.putText(frame,
                        text,
                        (40, y_pos),
                        FONT_HERSHEY_SIMPLEX,
                        SMALL_FONTSCALE,
                        WHITE,
                        THICK,
                        LINE_AA)

    def _draw_legend_box(self, frame: np.array) -> None:
        """draw pts of selected object onto frame

        Args:
            frame (np.array): image of current frame
            zone_count (List[float]): object count, likely people, of each zone used
            in the zone analytics
        """
        assert self.legend_height is not None
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (self.legend_left_x, self.legend_starting_y),
                      (self.legend_left_x + 150,
                       self.legend_starting_y + self.legend_height),
                      BLACK,
                      FILLED)
        # apply the overlay
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    @staticmethod
    def _get_legend_height(inputs: Dict[str, Any], items: List[str]) -> int:
        """Get height of legend box needed to contain all items drawn"""
        no_of_items = len(items)
        if "zone_count" in items:
            # increase the number of items according to number of zones
            no_of_items += len(inputs["zone_count"])
        return 12 * no_of_items + 8 * (no_of_items - 1) + 20

    def _get_legend_registry(self) -> Dict[str, Any]:
        """Get registry of functions that draw items
        available in the legend"""
        return {
            'fps': self._draw_fps,
            'count': self._draw_count,
            'zone_count': self._draw_zone_count
        }

    def _set_legend_variables(self, position: str) -> None:
        assert self.legend_height != 0
        if position == "top":
            self.legend_starting_y = 10
        else:
            _, image_height = get_image_size(self.frame)
            self.legend_starting_y = image_height - 10 - self.legend_height

    def add_register(self, name: str, method: Any) -> None:
        """Add new legend drawing information to the registry

        Args:
            name (str): name of method, corresponding to key to get input
            method (Any): function of the method. Note that take in 1) image frame
            and 2) input needed for the method, taken in using inputs[<key>]
        """
        self.func_reg[name] = method
