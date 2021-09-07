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

from typing import Dict, Any
from peekingduck.pipeline.nodes.node import AbstractNode
from .mtcnnv1 import mtcnn_model


class Node(AbstractNode):
    
    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = mtcnn_model.MtcnnModel(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        bboxes, scores, landmarks, classes = self.model.predict(inputs["img"])
        outputs = {"bboxes": bboxes, "bbox_scores": scores, "landmarks": landmarks, 
                   "bbox_labels": classes}
        return outputs
