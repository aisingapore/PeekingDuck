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
Fast Pose Estimation model
"""

from typing import Dict, Any
from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.model.posenetv1 import posenet_model


class Node(AbstractNode):
    """PoseNet node class that initalises a PoseNet model to detect poses from
    an image

    PersonLab: Person Pose Estimation and Instance Segmentation with a Bottom-Up,
    Part-Based, Geometric Embedding Model
    https://arxiv.org/abs/1803.08225
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)
        self.model = posenet_model.PoseNetModel(config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """function that reads the image input and returns the bboxes
        of the specified objects chosen to be detected

        Args:
            inputs (dict): Dict with keys "img".

        Returns:
            outputs (dict): Dict with keys "bboxes", "keypoints", "keypoint_scores",
            "keypoint_conns"
        """
        bboxes, keypoints, keypoint_scores, keypoint_conns = self.model.predict(
            inputs["img"])
        outputs = {"bboxes": bboxes,
                   "keypoints": keypoints,
                   "keypoint_scores": keypoint_scores,
                   "keypoint_conns": keypoint_conns}
        return outputs
