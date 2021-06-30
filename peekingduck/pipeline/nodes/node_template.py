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
Node template for creating custom nodes
"""

from typing import Any, Dict
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """This is a template class of how to write a node for peekingduck."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)

        # initialize/load any configs and models here
        # self.config1 = config["config1"]
        # self.logger.info("model loaded with configs: %s", configs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """ This node does ___.

        Args:
            inputs (dict): Dict with keys "__", "__".

        Returns:
            outputs (dict): Dict with keys "__".
        """

        # result = do_something(inputs["in1"], inputs["in2"])
        # outputs = {"out1": result}
        # return outputs
