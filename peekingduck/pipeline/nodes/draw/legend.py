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
Displays info from dabble nodes such as fps, object count and zone counts in a legend box
"""

from typing import Any, Dict, List
from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.draw.utils.legend import Legend


class Node(AbstractNode):
    """Draw node for drawing Legend box and info on image

    The draw legend node dynamically pulls the output results of previous nodes
    And uses it to draw the information into a legend box. Currently draws fps,
    object counts and object count in zones.

    all_legend_item: config is all possible items that can be drawn in legend box
    include: is used to select which information would be drawn
    This is so we can have the outputs but choose not to drawn on screen

    Inputs:

        all (:obj:`Any`): Receives inputs from all preceding outputs to use
        ass dynamic input for legend creation.

    Outputs:
        |none|

    Configs:
        position (:obj:`str`): **default = "bottom"**
            Position to draw legend box. "top" draws it at the top-left position
            while "bottom" draws it at bottom-left.

        include (:obj:`list`): **default = ["all_legend_items"]**
            List of information to draw. Current can draw "fps", "count" and/or
            "zone_count". The default value "all_legend_items" draws everything
            dynamically depending on inputs.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.legend_items: List[str] = []

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Draws legend box with information from nodes

        Args:
            inputs (dict): Dict with all available keys

        Returns:
            outputs (dict): Dict with keys "none"
        """
        if len(self.legend_items) == 0:
            # Check inputs to set legend items to draw
            if self.include[0] == 'all_legend_items': # type: ignore
                self.include = self.all_legend_items # type: ignore
            self._include(inputs)
        if len(self.legend_items) != 0:
            Legend().draw(inputs, self.legend_items, self.position)  # type: ignore
        else:
            return {}
        # cv2 weighted does not update the referenced image. Need to return and replace.
        return {'img': inputs['img']}

    def _include(self, inputs: Dict[str, Any]) -> None:
        for item in self.all_legend_items:
            if item in inputs.keys() and item in self.include:
                self.legend_items.append(item)
