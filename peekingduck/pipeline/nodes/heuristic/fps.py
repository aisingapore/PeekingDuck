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
Calculates the FPS of video
"""

from typing import Any, Dict
from time import perf_counter
import numpy as np

from peekingduck.pipeline.nodes.node import AbstractNode

NUM_FRAMES = 14


class Node(AbstractNode):
    """ FPS node class that calculates the FPS of the image frame """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)
        self.time_window = [float(0)]

        self.moving_avg_thres = config["moving_avg"]
        self.moving_average_fps = []
        self.global_avg_fps = 0
        self.count = 0

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """ Calculates FPS using the time difference between the current
        frame and the previous frame.

        Args:
            inputs: ["pipeline_end"]

        Returns:
            outputs (dict): Dict with key "fps".
        """

        if len(self.time_window) > NUM_FRAMES:
            self.time_window.pop(0)

        self.time_window.append(perf_counter())

        num_frames = len(self.time_window)
        time_diff = self.time_window[-1] - self.time_window[0]
        average_fps = num_frames / time_diff

        # Logs 14-frame moving average per frame
        if self.moving_avg_thres:
            moving_average = self._moving_average(average_fps)
            self.logger.info('14-frame Moving Average FPS: %s', moving_average)

        # Calculate global cumulative moving average
        self.global_avg_fps = self._global_average(average_fps)
        # Log global cumulative average when pipeline ends
        if inputs["pipeline_end"]:
            self.logger.info('Approximate Global Average FPS: %s',
                self.global_avg_fps)

        return {"fps": average_fps}

    def _moving_average(self, average_fps):
        self.moving_average_fps.append(average_fps)
        if len(self.moving_average_fps) > NUM_FRAMES:
            self.moving_average_fps.pop(0)
        moving_average_val = np.mean(self.moving_average_fps)
        return round(moving_average_val, 1)

    def _global_average(self, average_fps):
        # Cumulative moving average formula
        global_average = \
            (average_fps + self.count*self.global_avg_fps) / (self.count + 1)
        self.count += 1
        return round(global_average, 1)
