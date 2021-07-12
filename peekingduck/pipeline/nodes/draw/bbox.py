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
Draw bounding boxes over detected object
"""

from typing import Any, Dict
from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.draw.utils.bbox import draw_bboxes
from peekingduck.pipeline.nodes.draw.utils.constants import CHAMPAGNE


class Node(AbstractNode):
    """Draw node for drawing bboxes onto image"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)
        self.show_labels = config['show_labels']

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.show_labels:
            draw_bboxes(inputs["img"], inputs["bboxes"],
                        CHAMPAGNE, inputs["bbox_labels"])  # type: ignore
        else:
            draw_bboxes(inputs["img"], inputs["bboxes"], CHAMPAGNE)  # type: ignore
        return {}
