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
Slower, accurate Pose Estimation model. Requires a object detector
"""


from typing import Dict, Any
from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.model.hrnetv1 import hrnet_model


class Node(AbstractNode):
    """HRNet node class that initialises and use hrnet model to infer poses
    from detected bboxes

    Deep High-Resolution Representation Learning for Visual Recognition
    https://arxiv.org/abs/1908.07919
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)
        self.model = hrnet_model.HRNetModel(config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """function that reads the bbox input and returns the poses
        and pose bbox of the specified objects chosen to be detected

        Args:
            inputs (dict): Dict with keys "img", "bboxes"

        Returns:
            outputs (dict): Dict with keys "keypoints", "keypoint_scores", "keypoint_conns"
        """
        keypoints, keypoint_scores, keypoint_conns = self.model.predict(
            inputs["img"], inputs["bboxes"])

        outputs = {"keypoints": keypoints,
                   "keypoint_scores": keypoint_scores,
                   "keypoint_conns": keypoint_conns}
        return outputs
