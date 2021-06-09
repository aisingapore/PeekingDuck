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

from typing import Dict, Any
from peekingduck.pipeline.nodes.node import AbstractNode
from .hrnetv1 import hrnet_model


class Node(AbstractNode):
    """HRNet node class that initialises and use hrnet model to infer poses
    from detected bboxes
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)
        self.model = hrnet_model.HRNetModel(config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """function that reads the bbox input and returns the poses
        and pose bbox of the specified objects chosen to be detected

        Args:
            inputs (Dict): Dictionary of inputs with keys "img", "bboxes"

        Returns:
            outputs (Dict): keypoints output in dictionary format with keys
            "keypoints", "keypoint_scores", "keypoint_conns"
        """
        keypoints, keypoint_scores, keypoint_conns, _ = self.model.predict(
            inputs["img"], inputs["bboxes"])

        output = {"keypoints": keypoints,
                  "keypoint_scores": keypoint_scores,
                  "keypoint_conns": keypoint_conns}
        #   "bboxes": keypoint_bboxes}
        return output
