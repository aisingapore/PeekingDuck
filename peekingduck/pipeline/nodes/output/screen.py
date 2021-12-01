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
Shows the outputs on your display.
"""

from typing import Any, Dict

import cv2

from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """Streams the output on your display.

    Inputs:
        |img|

    Outputs:
        |pipeline_end|

    Configs:
        window_name (:obj:`str`): **default = "PeekingDuck"** |br|
            Name of the displayed window.
        window_width (:obj:`int`): **default = 1280** |br|
            Width of the displayed window, in pixels.
        window_height (:obj:`int`): **default = 720** |br|
            Height of the displayed window, in pixels.
        window_x_coord (:obj:`int`): **default = 0** |br|
            X-coordinate of the top left corner of the displayed window, with reference
            from the top left corner of the screen, in pixels.
        window_y_coord (:obj:`int`): **default = 0** |br|
            Y-coordinate of the top left corner of the displayed window, with reference
            from the top left corner of the screen, in pixels.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.window_name, self.window_x_coord, self.window_y_coord)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Show the outputs on your display"""

        cv2.imshow(self.window_name, inputs["img"])

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyWindow(self.window_name)
            return {"pipeline_end": True}

        return {"pipeline_end": False}
