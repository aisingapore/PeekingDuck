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
Draws a tag (from ``obj_attrs``) above each bounding box
"""

from collections import deque
from typing import Any, Dict, List, Union

from peekingduck.pipeline.nodes.draw.utils.bbox import draw_tags
from peekingduck.pipeline.nodes.draw.utils.constants import TOMATO
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """Draws a tag above each bounding box in the image, using information from selected attributes
    in ``obj_attrs``. In `Example 1` below, ``obj_attrs`` has 2 attributes (`<attr a>` and
    `<attr b>`). There are `x` detected bounding boxes, and each attribute has `x` corresponding
    tags stored in a list. The ``get`` config described subsequently is used to choose the
    attribute to be drawn.

    >>> # Example 1
    >>> {"obj_attrs": {<attr a>: [<tag 1>, ..., <tag x>], <attr b>: [<tag 1>, ..., <tag x>]}}

    The following type conventions need to be observed: |br|
    1. Each attribute must be of type :obj:`List`, e.g. ``<attr a>: [<tag 1>, ..., <tag x>]`` |br|
    2. Each tag within the list must be of type :obj:`str` or :obj:`int`

    In `Example 2` below, ``obj_attrs`` has 3 attributes (`"ids"`, `"gender"` and `"age"`), where
    the last 2 attributes are nested within `"details"`. There are 2 detected bounding boxes, and
    for the first one, possible tags to be drawn are `1`, `"female"` or `52`.

    >>> # Example 2
    >>> {"obj_attrs": {"ids":[1,2], "details": {"gender": ["female","male"], "age": [52,17]}}

    Inputs:
        |img|

        |bboxes|

        |obj_attrs|

    Outputs:
        |no_output|

    Configs:
        get (:obj:`List[str]`): **default = []**. |br|
            List the keys of the ``obj_attrs`` dictionary required to get the desired tag. For
            example 2, to draw the tag given by the attribute `"ids"`, the required key
            is `["ids"]`. To draw the tag given by the attribute `"age"`, as `"age"` is nested
            within `"details"`, the required keys are `["details", "age"]`.

    .. versionchanged:: 1.2.0 |br|
        :mod:`draw.tag` used to take in ``obj_tags`` (:obj:`List[str]`) as an input data type,
        which has been deprecated and now subsumed under ``obj_attrs``
        (:obj:`Dict[str, List[Any]]`), giving this node more flexibility.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Draws a tag above each bounding box.

        Args:
            inputs (dict): Dictionary with keys "bboxes", "obj_attrs", "img".

        Returns:
            outputs (dict): Dictionary with keys "none".
        """

        keys = deque(self.get)
        attribute = _get_value(inputs["obj_attrs"], keys)
        if type(attribute) != list:
            raise ValueError(
                f"The attribute of interest has to be of type 'list', containing a list of tags. "
                f"However, the attribute chosen here was: {attribute} which is of type: "
                f"{type(attribute)}."
            )
        # if empty list, nothing to draw
        if not attribute:
            return {}

        if type(attribute[0]) != int and type(attribute[0]) != str:
            raise ValueError(
                f"Each tag has to be of type 'int' or 'str'. "
                f"However, the first detected tag here was {attribute[0]} which is of type: "
                f"{type(attribute[0])}."
            )

        draw_tags(inputs["img"], inputs["bboxes"], attribute, TOMATO)

        return {}


def _get_value(data: Dict[str, Any], keys: deque) -> List[Union[str, int]]:
    if not keys:
        return data
    key = keys.popleft()
    return _get_value(data[key], keys)
