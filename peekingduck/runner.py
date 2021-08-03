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
Main engine for Peekingduck processes
"""

import copy
import sys
import logging
from typing import List
from peekingduck.pipeline.pipeline import Pipeline
from peekingduck.declarative_loader import DeclarativeLoader
from peekingduck.pipeline.nodes.node import AbstractNode


class Runner():
    """Runner class that uses the declared nodes to create pipeline to run inference
    """

    def __init__(self,
                 RUN_PATH: str = "",
                 config_updates_cli: str = None,
                 CUSTOM_NODE_PARENT_FOLDER: str = None,
                 nodes: List[AbstractNode] = None):
        """
        Args:
            RUN_PATH (str): path to yaml file of node pipeine declaration.
            config_updates_cli (str): stringified nested dictionaries of configs.
            CUSTOM_NODE_PARENT_FOLDER (str): parent folder of the custom nodes folder.
            nodes (:obj:'list' of :obj:'Node'): if not using declarations via yaml,
                initialize by giving the node stack as a list

        """

        self.logger = logging.getLogger(__name__)

        if not nodes and RUN_PATH:
            # create Graph to run
            self.node_loader = DeclarativeLoader(
                RUN_PATH, config_updates_cli, CUSTOM_NODE_PARENT_FOLDER)  # type: ignore

            self.pipeline = self.node_loader.get_pipeline()

        # If Runner given nodes, instantiated_nodes is created differently
        else:
            try:
                self.pipeline = Pipeline(nodes) # type: ignore
            except ValueError as error:
                self.logger.error(str(error))
                sys.exit(1)

    def run(self) -> None:
        """execute single or continuous inference
        """
        while not self.pipeline.terminate:
            for node in self.pipeline.nodes:
                if "pipeline_end" in self.pipeline.data and \
                        self.pipeline.data["pipeline_end"]:  # type: ignore

                    self.pipeline.terminate = True
                    if "pipeline_end" not in node.inputs:
                        continue

                if "all" in node.inputs:
                    inputs = copy.deepcopy(self.pipeline.data)
                else:
                    inputs = {key: self.pipeline.data[key]
                              for key in node.inputs if key in self.pipeline.data}

                outputs = node.run(inputs)
                self.pipeline.data.update(outputs)  # type: ignore

    def get_run_config(self) -> List[str]:
        """retrieve run configs

        Returns:
            Dict[Any]: run configs being used for runner
        """
        return self.node_loader.node_list
