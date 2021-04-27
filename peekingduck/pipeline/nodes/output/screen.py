import sys
import cv2
from peekingduck.pipeline.nodes.node import AbstractNode

class Node(AbstractNode):
    def __init__(self, config):
        super().__init__(config, node_path=__name__)

    def run(self, inputs: dict):
        cv2.imshow('livefeed', inputs[self.inputs[0]])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit(1)
        return {}
