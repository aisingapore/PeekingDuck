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
Displays selected information from preceding nodes in a legend box.
"""

from typing import Any, Dict, List

from peekingduck.pipeline.nodes.draw.utils.legend import Legend
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """Draws a translucent legend box on a corner of the image, containing selected information
    produced by preceding nodes in the format ``<data type>: <value>``. Supports in-built
    PeekingDuck data types defined in :doc:`Glossary </glossary>` as well as custom data types
    produced by custom nodes.

    This example screenshot shows ``fps`` from :mod:`dabble.fps`, ``count`` from
    :mod:`dabble.bbox_count` and ``avg`` from :mod:`dabble.statistics` displayed within the legend
    box. Note that values of float type such as ``fps`` and ``avg`` are displayed in 2 decimal
    places.

    .. image:: /assets/api/legend.png
    |br|

    Inputs:
        |all|

    Outputs:
        |img|

    Configs:
        all_legend_items (:obj:`List[str]`):
            **default = ["fps", "count", "zone_count", "avg", "min", "max"]**. |br|
            A list of all valid data types that can be processed by this node. To process custom
            data types produced by custom nodes, add the custom data type to this list. Note that
            to actually draw the information in the legend box, the data type has to be added to
            the list in the ``include`` config as well.

            .. versionchanged:: 1.2.0
                Permit adding of custom data types produced by custom nodes. Also added
                in-built PeekingDuck data types ``avg``, ``min``, ``max`` created from
                :mod:`dabble.statistics` to default list.

        position (:obj:`str`): **{"top", "bottom"}, default = "bottom"**. |br|
            Position to draw legend box. "top" draws it at the top-left position while "bottom"
            draws it at bottom-left.
        include (:obj:`List[str]`): **default = ["*"]**. |br|
            Include in this list the desired information to be drawn within the legend box, such
            as ``["fps", "count", "avg"]`` in the example screenshot. The default value is the
            wildcard ``*``, which draws all the information from ``all_legend_items`` as long as
            they were produced from preceding nodes. To draw information from custom data types
            produced by custom nodes, the data type has to be added both here as well as the list
            in the ``all_legend_items`` config.

            .. versionchanged:: 1.2.0
                Default value changed to wildcard character "*".
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.include: List[str]
        self.legend_items: List[str] = []

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Draws legend box with information from nodes.

        Args:
            inputs (dict): Dictionary with all available keys.

        Returns:
            outputs (dict): Dictionary with keys "none".
        """
        if len(self.legend_items) == 0:
            # Check inputs to set legend items to draw
            if self.include[0] == "*":
                self.include = self.all_legend_items
            self._include(inputs)
        if len(self.legend_items) != 0:
            Legend().draw(inputs, self.legend_items, self.position)
        else:
            return {}
        # cv2 weighted does not update the referenced image. Need to return and replace.
        return {"img": inputs["img"]}

    def _include(self, inputs: Dict[str, Any]) -> None:
        for item in self.all_legend_items:
            if item in inputs.keys() and item in self.include:
                self.legend_items.append(item)
