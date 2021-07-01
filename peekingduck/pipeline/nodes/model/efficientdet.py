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
Slower, accurate Object Detection model
"""


from typing import Dict, Any
from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.model.efficientdet_d04 import efficientdet_model


class Node(AbstractNode):
    """EfficientDet node class that initializes and uses efficientdet model to detect
    bounding boxes from an image.

    EfficientDet: Scalable and Efficient Object Detection
    https://arxiv.org/abs/1911.09070
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)
        self.model = efficientdet_model.EfficientDetModel(config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Function that takes an image as input and returns bboxes of objects specified
        in config.

        Args:
            inputs (Dict): Dict with key "img"

        Returns:
            outputs (Dict): Dict with keys "bboxes".
        """
        # Currently prototyped to return just the bounding boxes
        # without the scores
        bboxes, labels, scores = self.model.predict(inputs["img"])
        outputs = {"bboxes": bboxes, "bbox_labels": labels, "bbox_scores": scores}
        return outputs
