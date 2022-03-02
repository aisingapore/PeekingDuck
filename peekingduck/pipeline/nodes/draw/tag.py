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

import re
from typing import Any, Dict, List

from peekingduck.pipeline.nodes.draw.utils.bbox import draw_tags
from peekingduck.pipeline.nodes.draw.utils.constants import TOMATO
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """Draws a tag above each bounding box in the image, using information from selected attributes
    in ``obj_attrs``. In the general example below, ``obj_attrs`` has 2 attributes (`<attr a>` and
    `<attr b>`). There are `n` detected bounding boxes, and each attribute has `n` corresponding
    tags stored in a list. The ``show`` config described subsequently is used to choose the
    attribute or attributes to be drawn.

    >>> {"obj_attrs": {<attr a>: [<tag 1>, ..., <tag n>], <attr b>: [<tag 1>, ..., <tag n>]}}

    The following type conventions need to be observed:

    * Each attribute must be of type :obj:`List`, e.g. ``<attr a>: [<tag 1>, ..., <tag n>]`` \

    * Each tag must be of type :obj:`str`, :obj:`int`, :obj:`float` or :obj:`bool` to be \
    convertable into :obj:`str` type for drawing

    In the example below, ``obj_attrs`` has 3 attributes (`"ids"`, `"gender"` and `"age"`), where
    the last 2 attributes are nested within `"details"`. There are 2 detected bounding boxes, and
    thus each attribute consists of a list with 2 tags.

    >>> # Example
    >>> {"obj_attrs": {"ids":[1,2], "details": {"gender": ["female","male"], "age": [52,17]}}

    The table below illustrates how ``show`` can be configured to achieve different outcomes. Key
    takeaways are:

    * To draw nested attributes, include all the keys leading to them (within the ``obj_attrs`` \
    dictionary), separating each key with a ``->``. \

    * To draw multiple attributes above each bounding box, add them to the list of ``show`` \
    config. Attributes will be separated by ``, `` when drawn.

    +-----+-----------------------------------------+---------------+---------------+
    | No. | ``show`` config                         | Tag above 1st | Tag above 2nd |
    |     |                                         | bounding box  | bounding box  |
    +-----+-----------------------------------------+---------------+---------------+
    | 1.  | ["ids"]                                 | "1"           | "2"           |
    +-----+-----------------------------------------+---------------+---------------+
    | 2.  | ["details -> gender"]                   | "female"      | "male"        |
    +-----+-----------------------------------------+---------------+---------------+
    | 3.  | ["details -> age", "details -> gender"] | "52, female"  | "17, male"    |
    +-----+-----------------------------------------+---------------+---------------+

    Inputs:
        |img|

        |bboxes|

        |obj_attrs|

    Outputs:
        |no_output|

    Configs:
        show (:obj:`List[str]`): **default = []**. |br|
            List the desired attributes to be drawn. For more details on how to use this config,
            see the section above.

    .. versionchanged:: 1.2.0 |br|
        :mod:`draw.tag` used to take in ``obj_tags`` (:obj:`List[str]`) as an input data type,
        which has been deprecated and now subsumed under ``obj_attrs`` (:obj:`Dict[str, Any]`),
        giving this node more flexibility.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.attr_keys = []
        if not self.show:
            raise KeyError(
                "The 'show' config is currently empty. Add the desired attributes to be drawn "
                "to the list, in order to proceed."
            )
        for attr in self.show:
            attr = re.sub(" ", "", attr)
            self.attr_keys.append(re.split(r"->", attr))

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Draws a tag above each bounding box.

        Args:
            inputs (dict): Dictionary with keys "bboxes", "obj_attrs", "img".

        Returns:
            outputs (dict): Dictionary with keys "none".
        """
        tags = self._tags_from_obj_attrs(inputs["obj_attrs"])
        # if empty list, nothing to draw
        if not tags:
            return {}
        draw_tags(inputs["img"], inputs["bboxes"], tags, TOMATO)

        return {}

    def _tags_from_obj_attrs(self, inputs: Dict[str, Any]) -> List[str]:
        """Process inputs from various attributes into tags for drawing."""
        all_attrs: List[List[Any]] = []
        for attr_key in self.attr_keys:
            attr = _deep_get_value(inputs, attr_key.copy())
            if not isinstance(attr, list):
                raise TypeError(
                    f"The attribute of interest has to be of type 'list', containing a list of "
                    f"tags. However, the attribute chosen here was: {attr} which is of type: "
                    f"{type(attr)}."
                )
            all_attrs.append(attr)

        tags = []
        for attr in list(zip(*all_attrs)):
            if attr:
                if (
                    not isinstance(attr[0], str)
                    and not isinstance(attr[0], int)
                    and not isinstance(attr[0], float)
                    and not isinstance(attr[0], bool)
                ):
                    raise TypeError(
                        f"A tag has to be of type 'str', 'int', 'float' or 'bool' to be "
                        f"convertable to a string. However, the tag: {attr[0]} is of type: "
                        f"{type(attr[0])}"
                    )
            attr_str = map(str, attr)
            tags.append(", ".join(attr_str))

        return tags


def _deep_get_value(data: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """Recursively goes through the keys of a dictionary to obtain the final value."""
    if not keys:
        return data
    key = keys.pop(0)
    return _deep_get_value(data[key], keys)
