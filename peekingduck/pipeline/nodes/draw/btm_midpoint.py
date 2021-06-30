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
Draws the lowest middle point of a bounding box
"""

from typing import Dict, Any
from peekingduck.pipeline.nodes.node import AbstractNode
from .utils.drawfunctions import draw_pts


class Node(AbstractNode):
    """This draw node draws the bottom midpoints"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Methods that draws btm midpoint of bounding bboxes

         Args:
             inputs (dict): Dict with keys "bboxes".

         Returns:
             outputs (dict): Empty dictionary.
         """
        draw_pts(inputs["img"],
                 inputs["btm_midpoint"])

        return {}
