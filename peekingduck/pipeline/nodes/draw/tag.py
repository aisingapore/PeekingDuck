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

from peekingduck.pipeline.nodes.node import AbstractNode
from .utils.drawfunctions import draw_tags


class Node(AbstractNode):
    def __init__(self, config: Dict) -> None:
        super().__init__(config, name='draw.tag')

    def run(self, inputs: Dict) -> Dict:
        """Draws a tag above each bounding box.

        Args:
            inputs: ["bboxes", "obj_tags", "img"]

        Returns:
            outputs: ["img"]

        """

        draw_tags(inputs["img"],
                  inputs["bboxes"],
                  inputs["obj_tags"])
                  
        return {}

