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
Shows the outputs on your display.
"""

from typing import Any, Dict

import cv2
import numpy as np
from peekingduck.pipeline.nodes.abstract_node import AbstractNode

MIN_DISPLAY_SIZE = 120


class Node(AbstractNode):
    """Streams the output on your display.

    Inputs:
        |img_data|

        |filename_data|

    Outputs:
        |pipeline_end_data|

    Configs:
        window_name (:obj:`str`): **default = "PeekingDuck"** |br|
            Name of the displayed window.
        window_size (:obj:`Dict`):
            **default = { do_resizing: False, width: 1280, height: 720 }** |br|
            Resizes the displayed window to the chosen width and weight, if
            ``do_resizing`` is set to ``true``. The size of the displayed
            window can also be adjusted by clicking and dragging.
        window_loc (:obj:`Dict`): **default = { x: 0, y: 0 }** |br|
            X and Y coordinates of the top left corner of the displayed window,
            with reference from the top left corner of the screen, in pixels.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.window_name, self.window_loc["x"], self.window_loc["y"])
        self.previous_filename = ""

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Show the outputs on your display"""
        img = inputs["img"]
        self._set_window_size(inputs["filename"], img)
        cv2.imshow(self.window_name, img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyWindow(self.window_name)
            return {"pipeline_end": True}

        return {"pipeline_end": False}

    def _set_window_size(self, current_filename: str, img: np.ndarray) -> None:
        """If `do_resizing` option is False, window size will be initialized to the image or
        video's frame default size for every new video or image.

        If `do_resizing` option is True, window size will be initialized to the config setting's
        width and height for every new video or image.

        The length of either sides of the display window will be clamped to a lower bound of
        `MIN_DISPLAY_SIZE`

        Args:
            current_filename (str): The filename from the `inputs` dictionary
            img (np.ndarray): The current image. The image will not be changed in this function.
        """
        if current_filename != self.previous_filename:
            # Initialize the window size for every new video
            if self.window_size["do_resizing"]:
                # Clamp the sides fo the window_size to have a minimum of MIN_DISPLAY_SIZE
                img_width = max(self.window_size["width"], MIN_DISPLAY_SIZE)
                img_height = max(self.window_size["height"], MIN_DISPLAY_SIZE)
            else:
                img_height, img_width, _ = img.shape
            cv2.resizeWindow(self.window_name, img_width, img_height)
            self.previous_filename = current_filename
        else:
            _, _, win_width, win_height = cv2.getWindowImageRect(self.window_name)
            if win_width < MIN_DISPLAY_SIZE or win_height < MIN_DISPLAY_SIZE:
                win_width = max(win_width, MIN_DISPLAY_SIZE)
                win_height = max(win_height, MIN_DISPLAY_SIZE)
                cv2.resizeWindow(self.window_name, win_width, win_height)
