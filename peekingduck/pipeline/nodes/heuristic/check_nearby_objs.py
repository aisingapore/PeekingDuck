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
Check if detected objects are near each other
"""

from typing import Dict, Any

import numpy as np

from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """Node that checks if any objects are near to each other"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)
        self.near_thres = config["near_threshold"]
        self.tag_msg = config["tag_msg"]

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Compares the 3D locations of all objects to see which objects are close to each other.
        If an object is close to another, tag it.

        Args:
            inputs (dict): Dict with keys "obj_3D_locs".

        Returns:
            outputs (dict): Dict with keys "obj_tags".
        """

        obj_tags = [""]*len(inputs["obj_3D_locs"])

        for idx_1, loc_1 in enumerate(inputs["obj_3D_locs"]):
            for idx_2, loc_2 in enumerate(inputs["obj_3D_locs"]):
                if idx_1 == idx_2:
                    continue

                dist_bet = np.linalg.norm(loc_1 - loc_2)
                if dist_bet < self.near_thres:
                    obj_tags[idx_1] = self.tag_msg
                    break

        return {"obj_tags": obj_tags}
