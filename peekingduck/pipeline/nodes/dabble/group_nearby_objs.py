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
Assigns objects in close proximity to groups.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.dabble.utils.quick_find import QuickFind


class Node(AbstractNode):
    """Groups objects that are near each other.

    It does so by comparing the 3D locations of all objects, and assigning
    objects near each other to the same group. The group associated with each
    object is accessed by the ``groups`` key of :term:`obj_attrs`.

    Inputs:
        |obj_3D_locs_data|

    Outputs:
        |obj_attrs_data|
        :mod:`dabble.group_nearby_objs` produces the ``groups`` attribute.

    Configs:
        obj_dist_threshold (:obj:`float`): **default = 1.5**. |br|
            Threshold of distance, in metres, between two objects. Objects with
            distance less than ``obj_dist_threshold`` would be assigned to the
            same group.

    .. versionchanged:: 1.2.0
        :mod:`draw.group_nearby_objs` used to return ``obj_tags``
        (:obj:`List[str]`) as an output data type, which has been deprecated
        and now subsumed under :term:`obj_attrs`. The same attribute is
        accessed by the ``groups`` key of :term:`obj_attrs`.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Checks the distance between 3d locations of a pair of objects.

        If distance is less than threshold, assign the objects to the same
        group. Repeat for all object pairs.
        """
        nearby_obj_pairs = self._find_nearby_obj_pairs(
            inputs["obj_3D_locs"], self.obj_dist_threshold
        )

        quickfind = QuickFind(len(inputs["obj_3D_locs"]))
        for (idx_1, idx_2) in nearby_obj_pairs:
            if not quickfind.connected(idx_1, idx_2):
                quickfind.union(idx_1, idx_2)

        return {"obj_attrs": {"groups": quickfind.get_group_alloc()}}

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {"obj_dist_threshold": float}

    @staticmethod
    def _find_nearby_obj_pairs(
        obj_locs: List[np.ndarray], obj_dist_threshold: float
    ) -> List[Tuple[int, int]]:
        """If the distance between 2 objects are less than the threshold,
        append their indexes to nearby_obj_pairs as a tuple.
        """
        nearby_obj_pairs = []
        for idx_1, loc_1 in enumerate(obj_locs):
            for idx_2, loc_2 in enumerate(obj_locs):
                if idx_1 == idx_2:
                    continue

                dist_bet = np.linalg.norm(loc_1 - loc_2)

                if dist_bet <= obj_dist_threshold:
                    if (idx_2, idx_1) not in nearby_obj_pairs:
                        nearby_obj_pairs.append((idx_1, idx_2))

        return nearby_obj_pairs
