from typing import List, Set
from peekingduck.pipeline.nodes.node import AbstractNode


class Pipe:
    """ Pipe class that stores nodes and manages the data information used during inference
    """

    def __init__(self, nodes: List[AbstractNode]):
        """
        Args:
            nodes (:obj:'list' of :obj:'Node'): node stack as declared for use in
                inference pipeline
        """
        self._check_pipe(nodes)
        self.nodes = nodes
        self._datas = {}
        self.video_end = False

    def execute(self) -> None:
        """ executes all node contained within the pipe
        """

        for node in self.nodes:
            inputs = {key: self._datas[key]
                      for key in node.inputs if key in self._datas}
            outputs = node.run(inputs)

            # _node_name should no longer be a private variable as used here
            if node._name == "input.recorded":
                if outputs['end']:
                    self.video_end = True
                    break
            self._datas.update(outputs)

    @staticmethod
    def _check_pipe(nodes: Set[AbstractNode]):
        # 1. Check the initial node is a source node
        # 2. Check every subsequent node utilizes something that will exist
        # if reached end, it is all valid

        data_pool = []

        if nodes[0].inputs[0] == 'source':
            data_pool.extend(nodes[0].outputs)

        for node in nodes[1:]:

            if all(item in data_pool for item in node.inputs):
                data_pool.extend(node.outputs)
            else:
                raise ValueError(
                    'Graph nodes do not form proper channel. Please check nodes.'
                )
