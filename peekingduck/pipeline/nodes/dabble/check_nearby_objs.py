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
Checks if detected objects are near each other.
"""

from typing import Any, Dict

import numpy as np

from peekingduck.pipeline.nodes.abstract_node import AbstractNode


class Node(AbstractNode):
    """Checks if any objects are near each other.

    It does so by comparing the 3D locations of all objects to see which ones
    are near each other. If the distance between two objects is below the
    minimum threshold, both would be flagged as near with ``tag_msg``. These
    flags can be accessed by the ``flags`` key of :term:`obj_attrs`.

    Inputs:
        |obj_3D_locs_data|

    Outputs:
        |obj_attrs_data|
        :mod:`dabble.check_nearby_objs` produces the ``flags`` attribute which
        contains either the ``tag_msg`` for objects that are near each other or
        an empty string for objects with no other objects nearby.

    Configs:
        near_threshold (:obj:`float`): **default = 2.0**. |br|
            Threshold of distance, in metres, between two objects. Objects with
            distance less than ``near_threshold`` would be considered as 'near'.
        tag_msg (:obj:`str`): **default = "TOO CLOSE!"**. |br|
            Tag to identify objects which are near others.

    .. versionchanged:: 1.2.0
        :mod:`draw.check_nearby_objs` used to return ``obj_tags``
        (:obj:`List[str]`) as an output data type, which has been deprecated
        and now subsumed under :term:`obj_attrs`. The same attribute is
        accessed by using the ``flags`` key of :term:`obj_attrs`.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Compares the 3D locations of all objects to see which objects are
        close to each other.

        If an object is close to another, tag it.
        """
        obj_flags = [""] * len(inputs["obj_3D_locs"])

        for idx_1, loc_1 in enumerate(inputs["obj_3D_locs"]):
            for idx_2, loc_2 in enumerate(inputs["obj_3D_locs"]):
                if idx_1 == idx_2:
                    continue

                dist_bet = np.linalg.norm(loc_1 - loc_2)
                if dist_bet < self.near_threshold:
                    obj_flags[idx_1] = self.tag_msg
                    break

        return {"obj_attrs": {"flags": obj_flags}}

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {"near_threshold": float, "tag_msg": str}
