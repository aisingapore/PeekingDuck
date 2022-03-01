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

from typing import Any, Dict

from peekingduck.pipeline.nodes.draw.utils.legend import Legend
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """Draws a translucent legend box on a corner of the image, containing selected information
    produced by preceding nodes in the format ``<data type>: <value>``. Supports in-built
    PeekingDuck data types defined in :doc:`Glossary </glossary>` as well as custom data types
    produced by custom nodes.

    This example screenshot shows ``fps`` from :mod:`dabble.fps`, ``count`` from
    :mod:`dabble.bbox_count` and ``avg`` from :mod:`dabble.statistics` displayed within the legend
    box.

    .. image:: /assets/api/legend.png
    |br|

    Supported types that can be drawn are :obj:`int`, :obj:`float` and :obj:`str`.
    Note that values of float type such as ``fps`` and ``avg`` are displayed in 2 decimal places.

    Inputs:
        |all_input|

    Outputs:
        |img|

    Configs:
        position (:obj:`str`): **{"top", "bottom"}, default = "bottom"**. |br|
            Position to draw legend box. "top" draws it at the top-left position while "bottom"
            draws it at bottom-left.
        show (:obj:`List[str]`): **default = []**. |br|
            Include in this list the desired data type(s) to be drawn within the legend box, such
            as ``["fps", "count", "avg"]`` in the example screenshot. Custom data types produced
            by custom nodes are also supported.

    .. versionchanged:: 1.2.0
        Merged previous ``all_legend_items`` and ``include`` configs into a single ``show`` config
        for greater clarity. Added support for drawing custom data types produced by custom nodes,
        to improve the flexibility of this node.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Draws legend box with information from nodes.

        Args:
            inputs (dict): Dictionary with all available keys.

        Returns:
            outputs (dict): Dictionary with keys "none".
        """
        _check_data_type(inputs, self.show)
        if self.show:
            Legend().draw(inputs, self.show, self.position)
        # cv2 weighted does not update the referenced image. Need to return and replace.
        return {"img": inputs["img"]}


def _check_data_type(inputs, show):
    for item in show:
        if item not in inputs:
            raise KeyError(
                f"{item} was selected for drawing, but is not a valid data type from preceding "
                f"nodes."
            )
