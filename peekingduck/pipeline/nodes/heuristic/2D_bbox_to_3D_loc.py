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
from typing import Dict, Any

from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)

        self.height_factor = config['height_factor']
        self.focal_length = config['focal_length']

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Converts 2D bounding boxes into 3D locations.

        Args:
            inputs (dict): Dict with keys "bboxes".

        Returns:
            outputs (dict): Dict with keys "obj_3D_locs".
        """

        locations = []

        for idx, bbox in enumerate(inputs["bboxes"]):
            # Subtraction is to make the camera the origin of the coordinate system
            center_2d = (bbox[0:2] + bbox[2:4] * 0.5) - np.array([0.5, 0.5])

            z = (self.focal_length * self.height_factor) / bbox[3]
            x = (center_2d[0] * self.height_factor) / bbox[3]
            y = (center_2d[1] * self.height_factor) / bbox[3]

            point = np.array([x, y, z])
            locations.append({"idx": idx, "3D_loc": point})

        outputs = {"obj_3D_locs": locations}

        return outputs
