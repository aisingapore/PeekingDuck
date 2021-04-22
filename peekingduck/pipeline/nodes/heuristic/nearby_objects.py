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

import numpy as np
from typing import Dict

from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    def __init__(self, config: Dict) -> None:
        super().__init__(config, name="heuristic.nearby_objects")
        self.near_thres = config["near_threshold"]
        self.tag_msg = config["tag_msg"]

    def run(self, inputs: Dict) -> Dict:
        """Compares the 3D locations of all objects to see which objects are close to each other.
        If an object is close to another, tag it.

        Args:
            inputs: ["obj_3D_locs"]

        Returns:
            outputs: ["obj_tags"]
        """

        objs_info = []

        for obj_1 in inputs["obj_3D_locs"]:
            obj_info = {"idx": obj_1["idx"], "tag": " "}

            for obj_2 in inputs["obj_3D_locs"]:
                if obj_1 == obj_2:
                    continue

                dist_bet = np.linalg.norm(obj_1["3D_loc"] - obj_1["3D_loc"])
                if dist_bet < self.near_thres:
                    obj_info["tag"] = self.tag_msg
                    break
            objs_info.append(obj_info)

        outputs = {"obj_tags": objs_info}

        return outputs
