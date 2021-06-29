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
Displays the FPS of video
"""

from typing import Any, Dict
from time import perf_counter

from peekingduck.pipeline.nodes.node import AbstractNode
from .utils.drawfunctions import draw_fps

NUM_FRAMES = 14


class Node(AbstractNode):
    """ FPS node class that calculates the FPS and draw the FPS onto the image
    frame
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)

        self.time_window = [float(0)]

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """ Calculates FPS using the time difference between the current frame
        and the previous frame. Calculated FPS is then draw onto image frame

        Args:
            inputs: ["img"]

        Returns:
            outputs: [None]
        """

        if len(self.time_window) > NUM_FRAMES:
            self.time_window.pop(0)

        self.time_window.append(perf_counter())

        num_frames = len(self.time_window)
        time_diff = self.time_window[-1] - self.time_window[0]
        average_fps = num_frames / time_diff

        draw_fps(inputs['img'], average_fps)

        return {}
