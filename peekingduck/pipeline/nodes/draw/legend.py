# Copyright 2022 AI Singapore
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

from typing import Any, Dict, List, Union

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.draw.utils.legend import Legend


class Node(AbstractNode):
    """Draws a translucent legend box on a corner of the image, containing
    selected information produced by preceding nodes in the format
    ``<data type>: <value>``. Supports in-built PeekingDuck data types defined
    in :doc:`Glossary </glossary>` as well as custom data types produced by
    custom nodes.

    This example screenshot shows :term:`fps` from :mod:`dabble.fps`,
    :term:`count` from :mod:`dabble.bbox_count` and :term:`cum_avg` from
    :mod:`dabble.statistics` displayed within the legend box.

    .. image:: /assets/api/legend.png

    |br|

    With the exception of the :term:`zone_count` data type from
    :mod:`dabble.zone_count`, all other selected in-built PeekingDuck data
    types or custom data types must be of types :obj:`int`, :obj:`float`, or
    :obj:`str`. Note that values of float type such as :term:`fps` and
    :term:`cum_avg` are displayed in 2 decimal places.

    Inputs:
        |all_input_data|

    Outputs:
        |img_data|

    Configs:
        box_opacity (:obj:`float`): **default = 0.3**. |br|
            Opacity of legend box background. A value of 0.0 causes the legend box background to be
            fully transparent, while a value of 1.0 causes it to be fully opaque.
        font (:obj:`Dict[str, Union[float, int]]`): **default = {size: 0.7, thickness: 2}.** |br|
            Size and thickness of font within legend box. Examples of visually acceptable options
            are: |br|
            720p video: {size: 0.7, thickness: 2} |br|
            1080p video: {size: 1.0, thickness: 3}
        position (:obj:`str`): **{"top", "bottom"}, default = "bottom"**. |br|
            Position to draw legend box. "top" draws it at the top-left position while "bottom"
            draws it at bottom-left.
        show (:obj:`List[str]`): **default = []**. |br|
            Include in this list the desired data type(s) to be drawn within the legend box, such
            as ``["fps", "count", "cum_avg"]`` in the example screenshot. Custom data types
            produced by custom nodes are also supported. If no data types are included, an error
            will be produced.

    .. versionchanged:: 1.2.0
        Merged previous ``all_legend_items`` and ``include`` configs into a
        single ``show`` config for greater clarity. Added support for drawing
        custom data types produced by custom nodes, to improve the flexibility
        of this node.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.legend = Legend(self.show, self.position, self.box_opacity, self.font)
        if not self.show:
            raise KeyError(
                "To display information in the legend box, at least one data type must be "
                "selected in the 'show' config."
            )

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Draws legend box with information from nodes.

        Args:
            inputs (dict): Dictionary with all available keys.

        Returns:
            outputs (dict): Dictionary with keys "none".
        """
        _check_data_type(inputs, self.show)
        self.legend.draw(inputs)
        # cv2 weighted does not update the referenced image. Need to return and replace.
        return {"img": inputs["img"]}

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns a dictionary which maps the node's config keys to their
        respective typing.
        """
        return {
            "box_opacity": float,
            "font": Dict[str, Union[float, int]],
            "font.size": float,
            "font.thickness": int,
            "position": str,
            "show": List[str],
        }


def _check_data_type(inputs: Dict[str, Any], show: List[str]) -> None:
    """Checks if the data types provided in show were produced from preceding nodes."""
    for item in show:
        if item not in inputs:
            raise KeyError(
                f"'{item}' was selected for drawing, but is not a valid data type from preceding "
                "nodes."
            )
