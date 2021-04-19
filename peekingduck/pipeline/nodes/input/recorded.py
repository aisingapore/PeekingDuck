from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.input.utils.read import VideoNoThread


class Node(AbstractNode):
    def __init__(self, config):
        super().__init__(config, name='input.recorded')

        resolution = config['recorded']['resolution']
        input_source = config['recorded']['input_source']
        mirror_image = config['recorded']['mirror_image']

        self.videocap = VideoNoThread(resolution, input_source, mirror_image)

    def run(self, inputs: dict):
        '''
        input: ["source"],
        output: ["img", "end"]
        '''
        success, img = self.videocap.read_frame()
        if success:
            outputs = {self.outputs[0]: img, self.outputs[1]: False}
        else:
            outputs = {self.outputs[0]: None, self.outputs[1]: True}
        return outputs
