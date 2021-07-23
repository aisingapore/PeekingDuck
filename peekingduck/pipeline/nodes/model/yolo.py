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
Fast Object Detection model
"""

from typing import Dict, Any
from peekingduck.pipeline.nodes.node import AbstractNode
from .yolov4 import yolo_model


class Node(AbstractNode):
    """Yolo node class that initialises and use yolo model to infer bboxes
    from image frame
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)
        self.model = yolo_model.YoloModel(config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """function that reads the image input and returns the bboxes
        of the specified objects chosen to be detected

        Args:
            inputs (Dict): Dictionary of inputs with key "img"

        Returns:
            outputs (Dict): bbox output in dictionary format with keys
            "bboxes", "bbox_labels" and "bbox_scores"
        """
        # Currently prototyped to return just the bounding boxes
        # without the scores
        bboxes, labels, scores = self.model.predict(inputs["img"])
        outputs = {"bboxes": bboxes, "bbox_labels": labels, "bbox_scores": scores}
        return outputs
