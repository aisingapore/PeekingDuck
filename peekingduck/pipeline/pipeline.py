# Copyright 2022 AI Singapore
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
Pipeline class that stores nodes and manages the data information used during
inference.
"""

import textwrap
from typing import Any, Dict, List

from peekingduck.pipeline.nodes.abstract_node import AbstractNode


class Pipeline:  # pylint: disable=too-few-public-methods
    """Pipeline class that stores nodes and manages flow of data used during
    inference.

    Args:
        nodes (:obj:`List[AbstractNode]`): List of initialized nodes for the
            pipeline to run through.
    """

    def __init__(self, nodes: List[AbstractNode]) -> None:
        self.nodes = nodes
        self._check_pipe(nodes)
        self.data = {}  # type: ignore
        self.terminate = False

    def get_pipeline_results(self) -> Dict[str, Any]:
        """Gets all results data from nodes in pipeline.

        Returns:
            (:obj:`Dict[str, Any]`): Dictionary of all results from nodes in
            the pipeline.
        """
        return self.data

    @staticmethod
    def _check_pipe(nodes: List[AbstractNode]) -> None:
        # 1. Check the initial node is a source node
        # 2. Check every subsequent node utilizes something that will exist
        # if reached end, it is all valid
        data_pool = []

        if nodes[0].inputs[0] == "none":
            data_pool.extend(nodes[0].outputs)
        if nodes[0].inputs[0] == "message":
            data_pool.extend(nodes[0].outputs)

        for node in nodes[1:]:
            if all(item in data_pool for item in node.inputs) or "all" in node.inputs:
                data_pool.extend(node.outputs)
            else:
                msg = textwrap.dedent(
                    f"""\
                    Nodes in this pipeline do not form a proper channel:
                    {node.name} requires these inputs: {node.inputs}
                    Data pool only has these outputs from previous nodes: {data_pool}
                    Note that nodes run sequentially, in the order specified in the config file.
                    """
                )
                raise ValueError(msg)
