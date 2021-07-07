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


import sys
import logging
from typing import List
from peekingduck.pipeline.pipeline import Pipeline
from peekingduck.loaders import DeclarativeLoader
from peekingduck.pipeline.nodes.node import AbstractNode


class Runner():
    """Runner class that uses the declared nodes to create pipeline to run inference
    """

    def __init__(self, RUN_PATH: str, CUSTOM_NODE_PARENT_FOLDER: str = None,
                 nodes: List[AbstractNode] = None):
        """
        Args:
            RUN_PATH (str): path to yaml file of node pipeine declaration.
            CUSTOM_NODE_PARENT_FOLDER (str): parent folder of the custom nodes folder.
            nodes (:obj:'list' of :obj:'Node'): if not using declarations via yaml,
                initialize by giving the node stack as a list

        """

        self.logger = logging.getLogger(__name__)

        if not nodes:

            # create Graph to run
            self.node_loader = DeclarativeLoader(
                RUN_PATH, CUSTOM_NODE_PARENT_FOLDER)  # type: ignore

            self.pipeline = self.node_loader.get_pipeline()

        # If Runner given nodes, instantiated_nodes is created differently
        else:
            try:
                self.pipeline = Pipeline(nodes)
            except ValueError as error:
                self.logger.error(str(error))
                sys.exit(1)

    def run(self) -> None:
        """execute single or continuous inference
        """
        while not self.pipeline.terminate:
            self.pipeline.execute()
        del self.pipeline

    def get_run_config(self) -> List[str]:
        """retrieve run configs

        Returns:
            Dict[Any]: run configs being used for runner
        """
        return self.node_loader.node_list
