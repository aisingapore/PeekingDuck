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

from peekingduck.pipeline.nodes.abstract_node import AbstractNode


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
        self.aspect_ratio_thres = 5e-3
        self.current_filename = None

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Show the outputs on your display"""
        img = inputs["img"]
        if not self.window_size["do_resizing"]:
            img_height, img_width, _ = img.shape
            if inputs["filename"] is not self.current_filename:
                # Initialize the window size for every new video
                cv2.resizeWindow(self.window_name, img_width, img_height)
                self.current_filename = inputs["filename"]
            else:
                # Check the current window size, resize it if aspect ratio is not the same as image
                aspect_ratio = img_width / img_height
                _, _, win_width, win_height = cv2.getWindowImageRect(self.window_name)
                if abs(win_width / win_height - aspect_ratio) > self.aspect_ratio_thres:
                    win_height = int(win_width // aspect_ratio)
                    cv2.resizeWindow(self.window_name, win_width, win_height)

        elif inputs["filename"] is not self.current_filename:
            # Initialize to config's width and height settings if `do_resizing` is True
            cv2.resizeWindow(
                self.window_name, self.window_size["width"], self.window_size["height"]
            )
            self.current_filename = inputs["filename"]

        cv2.imshow(self.window_name, img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyWindow(self.window_name)
            return {"pipeline_end": True}

        return {"pipeline_end": False}
