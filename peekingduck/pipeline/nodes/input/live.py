from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.input.utils.read import VideoThread


class Node(AbstractNode):
    def __init__(self, config):
        super().__init__(config, name='input.live')

        resolution = config['resolution']
        input_source = config['input_source']
        mirror_image = config['mirror_image']

        self.logger.info("TESTING")

        self.videocap = VideoThread(resolution, input_source, mirror_image)

    def run(self, inputs: dict):
        success, img = self.videocap.read_frame()
        if success:
            outputs = {self.outputs[0]: img}
            return outputs

        raise Exception("An issue has been encountered reading the Image")
