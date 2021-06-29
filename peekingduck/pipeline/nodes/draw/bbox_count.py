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
Displays the counts of detected objects
"""


from typing import Any, Dict
from peekingduck.pipeline.nodes.node import AbstractNode
from .utils.drawfunctions import draw_count


class Node(AbstractNode):
    """Node that draws object counting"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Method that draws the count on the top left corner of image

         Args:
             inputs (dict): Dict with keys "count" and "img".
         Returns:
             outputs (dict): None
         """
        draw_count(inputs["img"],
                   inputs["count"])

        return {}
