from typing import Dict
from peekingduck.pipeline.nodes.node import AbstractNode
from .yolo import yolo_model


class Node(AbstractNode):
    def __init__(self, config):
        super().__init__(config, node_name=__name__)
        self.model = yolo_model.YoloModel(config)

    def run(self, inputs: Dict):
        # Currently prototyped to return just the bounding boxes
        # without the scores
        results, _, _ = self.model.predict(inputs[self.inputs[0]])
        outputs = {self.outputs[0]: results}
        return outputs
