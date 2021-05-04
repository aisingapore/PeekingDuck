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
from collections import Counter

from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """This node checks which objects are in groups that have exceeded
    the group size threshold."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)
        self.group_size_thres = config["group_size_thres"]

    def run(self, inputs: Dict[str, List[int]]) -> Dict[str, List[bool]]:
        """ Checks which objects are in groups that have exceeded the
        group size threshold.

        Args:
            inputs (dict): Dict with keys "obj_groups".

        Returns:
            outputs (dict): Dict with keys "is_obj_in_large_grp".
        """

        is_obj_in_large_grp = []
        large_groups = self._find_large_groups(
            inputs["obj_groups"], self.group_size_thres)
        for obj_group in inputs["obj_groups"]:
            if obj_group in large_groups:
                is_obj_in_large_grp.append(True)
            else:
                is_obj_in_large_grp.append(False)

        return {"is_obj_in_large_grp": is_obj_in_large_grp}

    @staticmethod
    def _find_large_groups(group_alloc: List[int], group_size_thres: int) -> List[int]:
        """ Creates a list containing the group numbers of groups that have exceeded
        the size defined by group_size_thres.
        """

        group_counter = Counter(group_alloc)
        large_groups = [group for group in group_counter if
                        group_counter[group] > group_size_thres]

        return large_groups
