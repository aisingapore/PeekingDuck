from typing import Dict
from peekingduck.pipeline.nodes.node import AbstractNode
from .utils.drawfunctions import draw_human_bboxes


class Node(AbstractNode):
    def __init__(self, config):
        super().__init__(config, name='DrawBBox')

    def run(self, inputs: Dict):

        draw_human_bboxes(inputs[self.inputs[1]],
                          inputs[self.inputs[0]])
        return {}
