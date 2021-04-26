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

from typing import Dict
from time import perf_counter

from peekingduck.pipeline.nodes.node import AbstractNode

class Node(AbstractNode):
    def __init__(self, config: Dict) -> None:
        super().__init__(config, node_path=__name__)

        self.current_frame_time = 0
        self.previous_frame_time = 0

    def run(self, inputs: Dict):
        self.current_frame_time = perf_counter()

        current_fps = 1 / (self.current_frame_time - self.previous_frame_time)

        self.logger("FPS: " + current_fps)

        self.previous_frame_time = self.current_frame_time

        return {}


