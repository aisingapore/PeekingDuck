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
    """Draws a tag above each bounding box in the image. Information for the tag is obtained from
    ``obj_attrs``.

    TODO:
    1. Explain <attribute>: [<tag1>, <tag2>]. List of tags
    2. Have a ValueError check for list.
    3. Update ValueError check for str or int in utils/bbox.py
    4. Check that no error when nothing is detected

    >>> # Example
    >>> {"obj_attrs": {"ids":[1,2], "details": {"gender": ["female","male"], "age": [52,17]}}

    In this example, there are 2 detected objects/ bounding boxes. For the first object, possible
    ``<attribute>: <tag>`` combinations are ``"ids": 1``, ``"gender": "female"`` or ``"age": 52``.
    The ``get`` config described below is used to choose the tag to be drawn. Tag has to be of type
    :obj:`str` or :obj:`int`, such as `1`, `"female"` or `52` here.

    Inputs:
        |img|

        |bboxes|

        |obj_attrs|

    Outputs:
        |no_output|

    Configs:
        get (:obj:`List[str]`): **default = []**. |br|
            List the keys of the ``obj_attrs`` dictionary required to get the desired tag. From
            the above example, to draw the tag given by the attribute `"ids"`, the required key
            is `["ids"]`. To draw the tag given by the attribute `"age"`, as `"age"` is nested
            within `"details"`, the required keys are `["details", "age"]`.
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
        tag = _get_value(inputs["obj_attrs"], keys)
        # if type(tag) != int and type(tag) != str:
        #     raise ValueError(
        #         f"The tag selected here is: {tag} and of type: {type(tag)}. "
        #         f"Ensure that the config 'get' lists the correct keys pointing to the desired tag, "
        #         f"and that the tag is of type 'int' or 'str'."
        #     )

        draw_tags(inputs["img"], inputs["bboxes"], tag, TOMATO)

        return {}


def _get_value(data: Dict[str, Any], keys: deque) -> List[Union[str, int]]:
    if not keys:
        return data
    key = keys.popleft()
    return _get_value(data[key], keys)
