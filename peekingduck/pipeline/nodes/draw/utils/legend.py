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

from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from cv2 import FONT_HERSHEY_SIMPLEX, LINE_AA

from peekingduck.pipeline.nodes.draw.utils.constants import (
    BLACK,
    FILLED,
    PRIMARY_PALETTE,
    PRIMARY_PALETTE_LENGTH,
    WHITE,
)
from peekingduck.pipeline.nodes.draw.utils.general import get_image_size

ZONE_COUNTS_HEADING = "-ZONE COUNTS-"
LEGEND_LEFT_X = 15


class Legend:  # pylint: disable=too-many-instance-attributes, too-few-public-methods
    """Legend class that uses available info to draw legend box on frame"""

    def __init__(
        self,
        items: List[str],
        position: str,
        box_opacity: float,
        font: Dict[str, Union[float, int]],
    ) -> None:
        self.items = items  # list of items to be drawn in legend box
        self.position = position
        self.box_opacity = box_opacity
        self.font_size = font["size"]
        self.font_thickness = font["thickness"]

        self.frame = None
        self.legend_starting_y = 0
        self.legend_width = 0
        self.legend_height = 0
        self.item_height = self._get_text_size("")[1]
        self.item_padding = self.item_height // 2

    def draw(self, inputs: Dict[str, Any]) -> None:
        """Draw legends onto image

        Args:
            inputs (dict): dictionary of all available inputs for drawing the legend
        """
        self.frame = inputs["img"]

        # legend box has to be drawn first to be "behind" text
        self._update_legend_size(inputs)
        self._set_legend_starting_y()
        self._draw_legend_box(self.frame)
        y_pos = self.legend_starting_y + self.item_height

        for item in self.items:
            if item == "zone_count":
                self._draw_zone_count(self.frame, y_pos, inputs[item])
            else:
                self._draw_item_info(self.frame, y_pos, item, inputs[item])
            y_pos += self.item_height + self.item_padding

    def _update_legend_size(self, inputs: Dict[str, Any]) -> None:
        """Update the width and height of the legend box"""
        self.legend_height = self._get_legend_height(inputs)
        self.legend_width = max(self.legend_width, self._get_legend_width(inputs))

    def _get_legend_height(self, inputs: Dict[str, Any]) -> int:
        """Get height of legend box needed to contain all items drawn"""
        num_items = len(self.items)
        if "zone_count" in self.items:
            # increase the number of items according to number of zones
            num_items += len(inputs["zone_count"])
        return (self.item_height + self.item_padding) * num_items

    def _get_legend_width(self, inputs: Dict[str, Any]) -> int:
        """Get width of legend box needed to contain all items drawn"""
        max_width = 0
        for item in self.items:
            if item != "zone_count":
                max_width = max(max_width, self._get_item_width(item, inputs[item]))
            else:
                max_width = self._get_item_width(ZONE_COUNTS_HEADING, "")
                for i, count in enumerate(inputs[item]):
                    max_width = max(
                        max_width,
                        self._get_item_width(f"ZONE-{i+1}", count)
                        + self.item_padding
                        + self.item_height,
                    )

        return max_width + 2 * self.item_padding

    def _get_item_width(
        self,
        item_name: str,
        item_info: Union[int, float, str],
    ) -> int:
        """Get width of the text to be drawn. If item info is
        of float type, it will be displayed in 2 decimal places.

        Args:
            item_name (str): name of the legend item
            item_info: Union[int, float, str]: info contained by the legend item
        """
        if not isinstance(item_info, (int, float, str)):
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

        return self._get_text_size(text)[0]

    def _set_legend_starting_y(self) -> None:
        assert self.legend_height != 0
        if self.position == "top":
            self.legend_starting_y = self.item_padding
        else:
            _, image_height = get_image_size(self.frame)
            self.legend_starting_y = (
                image_height - self.item_padding - self.legend_height
            )

    def _draw_legend_box(self, frame: np.ndarray) -> None:
        """draw pts of selected object onto frame

        Args:
            frame (np.array): image of current frame
        """
        assert self.legend_height is not None
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (LEGEND_LEFT_X, self.legend_starting_y - self.item_padding),
            (
                LEGEND_LEFT_X + self.legend_width,
                self.legend_starting_y + self.legend_height,
            ),
            BLACK,
            FILLED,
        )
        # apply the overlay
        cv2.addWeighted(
            overlay, self.box_opacity, frame, 1 - self.box_opacity, 0, frame
        )

    def _draw_item_info(
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
        self._put_text(frame, text, (LEGEND_LEFT_X + self.item_padding, y_pos))

    def _draw_zone_count(
        self, frame: np.ndarray, y_pos: int, counts: List[int]
    ) -> None:
        """Draw zone counts of all zones onto frame image

        Args:
            frame (np.array): image of current frame
            y_pos (int): y position to draw the count info text
            counts (list): list of zone counts
        """
        self._put_text(
            frame, ZONE_COUNTS_HEADING, (LEGEND_LEFT_X + self.item_padding, y_pos)
        )
        for i, count in enumerate(counts):
            y_pos += self.item_height + self.item_padding
            cv2.rectangle(
                frame,
                (LEGEND_LEFT_X + self.item_padding, y_pos),
                (
                    LEGEND_LEFT_X + self.item_padding + self.item_height,
                    y_pos - self.item_height,
                ),
                PRIMARY_PALETTE[(i + 1) % PRIMARY_PALETTE_LENGTH],
                FILLED,
            )
            text = f" ZONE-{i+1}: {count}"
            self._put_text(
                frame,
                text,
                (LEGEND_LEFT_X + self.item_padding + self.item_height, y_pos),
            )

    def _get_text_size(self, text: str) -> Tuple[int, int]:
        """Wrapper around cv2.getTextSize method to reduce number of arguments in calls"""
        return cv2.getTextSize(
            text,
            FONT_HERSHEY_SIMPLEX,
            self.font_size,
            self.font_thickness,
        )[0]

    def _put_text(self, frame: np.ndarray, text: str, pos: Tuple[int, int]) -> None:
        """Wrapper around cv2.putText method to reduce number of arguments in calls"""
        cv2.putText(
            frame,
            text,
            pos,
            FONT_HERSHEY_SIMPLEX,
            self.font_size,
            WHITE,
            self.font_thickness,
            LINE_AA,
        )
