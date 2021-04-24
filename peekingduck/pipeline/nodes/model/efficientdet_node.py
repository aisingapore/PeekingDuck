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
from typing import Dict
from peekingduck.pipeline.nodes.node import AbstractNode
from .efficientdet import efficientdet_model


class Node(AbstractNode):
    """Node for EfficientDet
    """

    def __init__(self, config):
        super().__init__(config, name='EfficientDet')
        self.model = efficientdet_model.EfficientDetModel(config)

    def run(self, inputs: Dict):
        # Currently prototyped to return just the bounding boxes
        # without the scores
        results, _, _ = self.model.predict(inputs[self.inputs[0]])
        outputs = {self.outputs[0]: results}
        return outputs
