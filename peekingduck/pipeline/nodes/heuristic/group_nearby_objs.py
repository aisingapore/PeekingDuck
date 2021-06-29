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
Assign objects in close proximity to groups
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.heuristic.utils.quick_find import QuickFind


class Node(AbstractNode):
    """This node groups objects that are near to each other."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)

        self.obj_dist_thres = config["obj_dist_thres"]

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """ Checks the distance between 3d locations of a pair of objects.
        If distance is less than threshold, assign the objects to the same group.
        Repeat for all object pairs.

        Args:
            inputs (dict): Dict with keys "obj_3D_locs".

        Returns:
            outputs (dict): Dict with keys "obj_groups".
        """

        nearby_obj_pairs = self._find_nearby_obj_pairs(
            inputs["obj_3D_locs"], self.obj_dist_thres)

        quickfind = QuickFind(len(inputs["obj_3D_locs"]))
        for (idx_1, idx_2) in nearby_obj_pairs:
            if not quickfind.connected(idx_1, idx_2):
                quickfind.union(idx_1, idx_2)

        return {"obj_groups": quickfind.get_group_alloc()}

    @staticmethod
    def _find_nearby_obj_pairs(obj_locs: List[np.array],
                               obj_dist_thres: float) -> List[Tuple[int, int]]:
        """If the distance between 2 objects are less than the threshold, append their
        indexes to nearby_obj_pairs as a tuple."""

        nearby_obj_pairs = []
        for idx_1, loc_1 in enumerate(obj_locs):
            for idx_2, loc_2 in enumerate(obj_locs):
                if idx_1 == idx_2:
                    continue

                dist_bet = np.linalg.norm(loc_1 - loc_2)

                if dist_bet <= obj_dist_thres:
                    if (idx_2, idx_1) not in nearby_obj_pairs:
                        nearby_obj_pairs.append((idx_1, idx_2))

        return nearby_obj_pairs
