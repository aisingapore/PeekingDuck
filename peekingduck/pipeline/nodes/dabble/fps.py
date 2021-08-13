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

from typing import Any, Dict, List
from statistics import mean
from time import perf_counter

from peekingduck.pipeline.nodes.node import AbstractNode

NUM_FRAMES = 10


class Node(AbstractNode):
    """ FPS node class that calculates the FPS of the image frame.

    This node calculates instantaneous FPS and a 10 frame moving average
    FPS. A preferred output setting can be set via the config file.

    Inputs:
        |pipeline_end|

    Outputs:
        |fps|

    Configs:
        fps_log_display (:obj:`bool`): **default="False"**

            Enables logging of 10 frame moving average FPS during execution of peekingduck

        fps_log_freq (:obj:`int`): **default="100"**
            Frequency of logging moving average FPS every n frames

        dampen_fps (:obj:`bool`): **default="True"**
            Returns moving average FPS if True, instantaneous FPS if False

    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.count = 0
        self.global_avg_fps = 0.0
        self.prev_frame_timestamp = 0.0
        self.moving_average_fps: List[float] = []

        if self.fps_log_display:
            self.logger.info('Moving average of FPS will be logged every: %s frames',
                             self.fps_log_freq)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """ Calculates FPS using the time difference between the current
        frame and the previous frame."""

        curr_frame_timestamp = perf_counter()

        # Frame level FPS
        frame_fps = 1./(curr_frame_timestamp - self.prev_frame_timestamp)
        self.prev_frame_timestamp = curr_frame_timestamp

        # Calculate moving average FPS (dampen_fps)
        average_fps = self._moving_average(frame_fps)

        # Ignore FPS of final frame to avoid skewing final average
        if not inputs["pipeline_end"]:
            if self.fps_log_display:
                if self.count % self.fps_log_freq == 0 and self.count != 0:
                    self.logger.info(
                        'Avg FPS over last 10 frames: %.2f', average_fps)

            # Calculate FPS over all processed frames
            self.global_avg_fps = self._global_average(frame_fps)

        if inputs["pipeline_end"]:
            self.logger.info('Avg FPS over all processed frames: %.2f',
                             self.global_avg_fps)

        return {"fps": average_fps} if self.dampen_fps else {"fps": frame_fps}

    def _moving_average(self, frame_fps: float) -> float:
        self.moving_average_fps.append(frame_fps)
        if len(self.moving_average_fps) > NUM_FRAMES:
            self.moving_average_fps.pop(0)
        moving_average_val = mean(self.moving_average_fps)
        return moving_average_val

    def _global_average(self, frame_fps: float) -> float:
        # Cumulative moving average formula
        global_average = \
            (frame_fps + self.count*self.global_avg_fps) / (self.count + 1)
        self.count += 1
        return global_average
