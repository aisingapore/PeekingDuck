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
Draws keypoints on a detected pose.
"""

from typing import Any, Dict, List

from peekingduck.pipeline.nodes.draw.utils.pose import draw_human_poses
from peekingduck.pipeline.nodes.abstract_node import AbstractNode


class Node(AbstractNode):
    """Draws poses onto image.

    The :mod:`draw.poses` node uses the :term:`keypoints`,
    :term:`keypoint_scores`, and :term:`keypoint_conns` predictions from pose
    models to draw the human poses onto the image. For better understanding,
    check out the pose models such as :mod:`HRNet <model.hrnet>` and
    :mod:`PoseNet <model.posenet>`.

    Inputs:
        |img_data|

        |keypoints_data|

        |keypoint_scores_data|

        |keypoint_conns_data|

    Outputs:
        |none_output_data|

    Configs:
        None.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.keypoint_dot_radius: int
        self.keypoint_dot_color = tuple(self.config["keypoint_dot_color"])
        self.keypoint_connect_color = tuple(self.config["keypoint_connect_color"])

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Draws pose details onto input image.

        Args:
            inputs (dict): Dictionary with keys "img", "keypoints", and
                "keypoint_conns".
        """
        draw_human_poses(
            inputs["img"],
            inputs["keypoints"],
            inputs["keypoint_conns"],
            self.keypoint_dot_color,
            self.keypoint_connect_color,
            self.keypoint_dot_radius,
        )
        return {}

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {
            "keypoint_dot_color": List[int],
            "keypoint_connect_color": List[int],
            "keypoint_dot_radius": int,
        }
