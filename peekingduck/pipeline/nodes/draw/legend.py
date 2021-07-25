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

from typing import Any, Dict, List
from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.draw.utils.legend import Legend


class Node(AbstractNode):
    """Draw node for drawing Legend box and info on image"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)
        self.all_legend_items = config['all_legend_items']
        self.include: List[str] = config['include']
        self.position = config['position']
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
            if self.include[0] == 'all':
                self.include = self.all_legend_items
            self._include(inputs)
        if len(self.legend_items) != 0:
            Legend().draw(inputs, self.legend_items, self.position)  # type: ignore
        else:
            return {}
        return {'img': inputs['img']}

    def _include(self, inputs: Dict[str, Any]) -> None:
        for item in self.all_legend_items:
            if item in inputs.keys() and item in self.include:
                self.legend_items.append(item)
