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
from typing import List, Set, Dict, Any
from peekingduck.pipeline.nodes.node import AbstractNode


class Pipeline:
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

            if 'end' in outputs and outputs['end']:
                self.video_end = True
                break

            self._datas.update(outputs)

    def get_pipeline_results(self) -> Dict[str, Any]:
        """get all results data of nodes in pipeline

        Returns:
            Dict[Any]: Dictionary of all pipeline node results
        """
        return self._datas

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
