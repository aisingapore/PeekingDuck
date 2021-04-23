from typing import Dict
from peekingduck.pipeline.nodes.node import AbstractNode
from .yolov4 import yolo_model


class Node(AbstractNode):
    """Yolo node class that initialises and use yolo model to infer bboxes
    from image frame
    """
    def __init__(self, config):
        super().__init__(config, name='Yolo')
        self.model = yolo_model.YoloModel(config)

    def run(self, inputs: Dict):
        """function that reads the image input and returns the bboxes
        of the specified objects chosen to be detected

        Args:
            inputs (Dict): Dictionary of inputs

        Returns:
            outputs (Dict): bbox output in dictionary format
        """
        # Currently prototyped to return just the bounding boxes
        # without the scores
        results, _, _ = self.model.predict(inputs[self.inputs[0]])
        outputs = {self.outputs[0]: results}
        return outputs
