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

from peekingduck.pipeline.nodes.draw.utils.constants import BLACK, THICK
from typing import Any, Dict
from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.draw.utils.legend import Legend


class Node(AbstractNode):
    """Draw node for drawing Legend box and info on image"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)
        self.all_legend_items = config['all_legend_items']
        self.exclude = config['exclude']
        self.legend_items = []

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Draws legend box with information from nodes.

        Args:
            inputs (dict): Dict with all available keys.

        Returns:
            outputs (dict): Dict with keys "none".
        """
        if len(self.legend_items) == 0:
            # Check inputs to set legend items to draw
            self._include(inputs)
        Legend().draw(inputs, self.legend_items)  # type: ignore
        return {'img': inputs['img']}

    def _include(self, inputs: Dict[str, Any]) -> None:
        for item in self.all_legend_items:
            if item in inputs.keys() and item not in self.exclude:
                self.legend_items.append(item)