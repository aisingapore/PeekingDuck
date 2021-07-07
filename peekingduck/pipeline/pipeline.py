# Copyright 2021 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pipeline class that stores nodes and manages the data information used during inference
"""

import copy
import textwrap
from typing import List, Dict, Any
from peekingduck.pipeline.nodes.node import AbstractNode


class Pipeline:
    """ Pipe class that stores nodes and manages the data information used during inference
    """

    def __init__(self, nodes: List[AbstractNode]) -> None:
        """
        Args:
            nodes (:obj:'list' of :obj:'Node'): node stack as declared for use in
                inference pipeline
        """
        self.nodes = nodes
        self._check_pipe(nodes)
        self._data = {}  # type: ignore
        self.terminate = False

    def __del__(self) -> None:
        for node in self.nodes:
            del node

    def execute(self) -> None:
        """ executes all node contained within the pipeline
        """
        for node in self.nodes:
            if "pipeline_end" in self._data and self._data["pipeline_end"]:  # type: ignore
                self.terminate = True
                if "pipeline_end" not in node.inputs:
                    continue

            if "all" in node.inputs:
                inputs = copy.deepcopy(self._data)
            else:
                inputs = {key: self._data[key]
                          for key in node.inputs if key in self._data}

            outputs = node.run(inputs)
            self._data.update(outputs)  # type: ignore


    def get_pipeline_results(self) -> Dict[str, Any]:
        """get all results data of nodes in pipeline

        Returns:
            Dict[Any]: Dictionary of all pipeline node results
        """
        return self._data

    @staticmethod
    def _check_pipe(nodes: List[AbstractNode]) -> None:
        # 1. Check the initial node is a source node
        # 2. Check every subsequent node utilizes something that will exist
        # if reached end, it is all valid

        data_pool = []

        if nodes[0].inputs[0] == 'none':
            data_pool.extend(nodes[0].outputs)

        for node in nodes[1:]:

            if all(item in data_pool for item in node.inputs) or "all" in node.inputs:
                data_pool.extend(node.outputs)
            else:
                msg = textwrap.dedent(f"""\
                    Nodes in this pipeline do not form a proper channel:
                    {node.name} requires these inputs: {node.inputs}
                    Data pool only has these outputs from previous nodes: {data_pool}
                    Note that nodes run sequentially, in the order specified in the config file.""")
                raise ValueError(msg)
