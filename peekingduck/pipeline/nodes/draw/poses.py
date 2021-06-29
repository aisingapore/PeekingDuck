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
Draws keypoints on a detected pose
"""

from typing import Any, Dict
from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.draw.utils.drawfunctions import draw_human_poses


class Node(AbstractNode):
    """Node for drawing poses onto image"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)
        self.keypoint_dot_color = tuple(config["keypoint_dot_color"])
        self.keypoint_dot_radius = config["keypoint_dot_radius"]
        self.keypoint_connect_color = tuple(config["keypoint_connect_color"])
        self.keypoint_text_color = tuple(config["keypoint_text_color"])

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """function that draws pose details onto input image

        Args:
            inputs (dict): Dict with keys "img", "keypoints", "keypoint_conns"
        """
        draw_human_poses(inputs["img"],  # type: ignore
                         inputs["keypoints"],
                         inputs["keypoint_scores"],
                         inputs["keypoint_conns"],
                         self.keypoint_dot_color,  # type: ignore
                         self.keypoint_dot_radius,
                         self.keypoint_connect_color,  # type: ignore
                         self.keypoint_text_color)  # type: ignore
        return {}
