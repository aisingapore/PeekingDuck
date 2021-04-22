import logging

from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.datamap import DataMap

class Node(AbstractNode):
    def __init__(self, config):
        super().__init__(config, name='PeopleCount')
        self._valid_input_type = ['bboxes']
        self._output_type = ['peoplecount']

    def get_valid_input_type(self):
        return self._valid_input_type

    def get_output_type(self):
        return self._output_type

    def get_outputs(self, dmap:DataMap):
        results = len(dmap.get(self._valid_input_type[0]))
        dmap.set(self._output_type[0], self._node_name, results)
        self.logger.info('peoplecount: {}'.format(results))
        return dmap
