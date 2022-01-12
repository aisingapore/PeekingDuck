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
Counts the number of detected boxes.
"""

from typing import Any, Dict

from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """Counts total number of detected objects.

    Inputs:
        |bboxes|

    Outputs:
        |count|

    Configs:
        None.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.avg = 0
        self.min = 0
        self.max = 0
        self.num_iter = 0
        self.mode = self.config["mode"]
        self.primary_attribute = self.config["attribute"]["primary_key"]
        self.secondary_attribute = self.config["attribute"]["secondary_key"]
        self.condition = self.config["condition"]

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Counts bboxes of object chosen in the frame.

        Note that this method requires that the bboxes returned to all belong
        to the same object category (for example, all "person").
        """
        self.num_iter += 1

        if not self.secondary_attribute:
            frame_info = inputs[self.primary_attribute]
        else:
            frame_info = inputs[self.primary_attribute][self.secondary_attribute]

        if self.mode == "len":
            count = len(frame_info)
        elif self.mode == "conditional_count":
            count = frame_info.count(self.condition)
        else:
            count = frame_info

        if count < self.min:
            self.min = count
        if count > self.max:
            self.max = count
        self.avg = (self.avg * self.num_iter + count) / (self.num_iter + 1)

        if not self.num_iter % 10:
            print(f"avg: {self.avg:.2f}, min: {self.min}, max: {self.max}")

        return {"avg": self.avg, "min": self.min, "max": self.max}
