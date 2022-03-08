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
functions for drawing legend related UI components
"""

from typing import Any, Dict, List, Union

import cv2
import numpy as np
from cv2 import FONT_HERSHEY_SIMPLEX, LINE_AA

from peekingduck.pipeline.nodes.draw.utils.constants import (
    BLACK,
    FILLED,
    PRIMARY_PALETTE,
    PRIMARY_PALETTE_LENGTH,
    SMALL_FONTSCALE,
    THICK,
    WHITE,
)
from peekingduck.pipeline.nodes.draw.utils.general import get_image_size


class Legend:
    """Legend class that uses available info to draw legend box on frame"""

    def __init__(self) -> None:
        self.legend_left_x = 15

        self.frame = None
        self.legend_starting_y = 0
        self.delta_y = 0
        self.legend_height = 0

    def draw(
        self, inputs: Dict[str, Any], items: List[str], position: str
    ) -> np.ndarray:
        """Draw legends onto image

        Args:
            inputs (dict): dictionary of all available inputs for drawing the legend
            items (list): list of items to be drawn in legend
            position (str): used to control whether legend box is drawn on top or bottom
        """
        self.frame = inputs["img"]

        self.legend_height = self._get_legend_height(inputs, items)
        self._set_legend_variables(position)

        self._draw_legend_box(self.frame)
        y_pos = self.legend_starting_y + 20
        for item in items:
            if item == "zone_count":
                self._draw_zone_count(self.frame, y_pos, inputs[item])
            else:
                self.draw_item_info(self.frame, y_pos, item, inputs[item])
            y_pos += 20

    def draw_item_info(
        self,
        frame: np.ndarray,
        y_pos: int,
        item_name: str,
        item_info: Union[int, float, str],
    ) -> None:
        """Draw item name followed by item info onto frame. If item info is
        of float type, it will be displayed in 2 decimal places.

        Args:
            frame (np.array): image of current frame
            y_pos (int): y_position to draw the count text
            item_name (str): name of the legend item
            item_info: Union[int, float, str]: info contained by the legend item
        """
        if isinstance(item_info, (int, float, str)):
            pass
        else:
            raise TypeError(
                f"With the exception of the 'zone_count' data type, "
                f"the draw.legend node only draws values that are of type 'int', 'float' or 'str' "
                f"within the legend box. The value: {item_info} from the data type: {item_name} "
                f"is of type: {type(item_info)} and is unable to be drawn."
            )

        if isinstance(item_info, float):
            text = f"{item_name.upper()}: {item_info:.2f}"
        else:
            text = f"{item_name.upper()}: {str(item_info)}"
        cv2.putText(
            frame,
            text,
            (self.legend_left_x + 10, y_pos),
            FONT_HERSHEY_SIMPLEX,
            SMALL_FONTSCALE,
            WHITE,
            THICK,
            LINE_AA,
        )

    def _draw_zone_count(
        self, frame: np.ndarray, y_pos: int, counts: List[int]
    ) -> None:
        """Draw zone counts of all zones onto frame image

        Args:
            frame (np.array): image of current frame
            y_pos (int): y position to draw the count info text
            counts (list): list of zone counts
        """
        text = "-ZONE COUNTS-"
        cv2.putText(
            frame,
            text,
            (self.legend_left_x + 10, y_pos),
            FONT_HERSHEY_SIMPLEX,
            SMALL_FONTSCALE,
            WHITE,
            THICK,
            LINE_AA,
        )
        for i, count in enumerate(counts):
            y_pos += 20
            cv2.rectangle(
                frame,
                (self.legend_left_x + 10, y_pos - 14),
                (self.legend_left_x + 30, y_pos + 5),
                PRIMARY_PALETTE[(i + 1) % PRIMARY_PALETTE_LENGTH],
                FILLED,
            )
            text = f" ZONE-{i+1}: {count}"
            cv2.putText(
                frame,
                text,
                (40, y_pos),
                FONT_HERSHEY_SIMPLEX,
                SMALL_FONTSCALE,
                WHITE,
                THICK,
                LINE_AA,
            )

    def _draw_legend_box(self, frame: np.ndarray) -> None:
        """draw pts of selected object onto frame

        Args:
            frame (np.array): image of current frame
            zone_count (List[float]): object count, likely people, of each zone used
            in the zone analytics
        """
        assert self.legend_height is not None
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (self.legend_left_x, self.legend_starting_y),
            (self.legend_left_x + 150, self.legend_starting_y + self.legend_height),
            BLACK,
            FILLED,
        )
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

    def _set_legend_variables(self, position: str) -> None:
        assert self.legend_height != 0
        if position == "top":
            self.legend_starting_y = 10
        else:
            _, image_height = get_image_size(self.frame)
            self.legend_starting_y = image_height - 10 - self.legend_height
