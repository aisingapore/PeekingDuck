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
Checks if number of objects in a group exceeds a threshold.
"""

from collections import Counter
from typing import Any, Dict, List

from peekingduck.pipeline.nodes.abstract_node import AbstractNode


class Node(AbstractNode):
    """Checks which groups have exceeded the group size threshold. The group
    associated with each object is accessed by the ``groups`` key of
    :term:`obj_attrs`.

    Inputs:
        |obj_attrs_data|
        :mod:`dabble.check_large_groups` requires the ``groups`` attribute.

    Outputs:
        |large_groups_data|

    Configs:
        group_size_threshold (:obj:`int`): **default = 5**. |br|
            Threshold of group size.

    .. versionchanged:: 1.2.0
        :mod:`draw.check_large_groups` used to take in ``obj_tags``
        (:obj:`List[str]`) as an input data type, which has been deprecated and
        now subsumed under :term:`obj_attrs`. The same attribute is accessed by
        using the ``groups`` key of :term:`obj_attrs`.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, List[int]]:
        """Checks which groups have exceeded the group size threshold,
        and returns a list of such groups.
        """
        group_counter: Counter = Counter(inputs["obj_attrs"]["groups"])
        large_groups = [
            group
            for group in group_counter
            if group_counter[group] > self.group_size_threshold
        ]

        return {"large_groups": large_groups}
