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
Main engine for PeekingDuck processes.
"""

import copy
import logging
import sys
from pathlib import Path
from typing import List

from peekingduck.declarative_loader import DeclarativeLoader, NodeList
from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.pipeline import Pipeline
from peekingduck.utils.requirement_checker import RequirementChecker


class Runner:
    """The runner class for creation of pipeline using declared/given nodes.

    The runner class uses the provided configurations to setup a node pipeline
    which is used to run inference.

    Args:
        run_config_path (:obj:`pathlib.Path` | :obj:`None`): If a path to
            *run_config.yml* is provided, uses
            :py:class:`DeclarativeLoader <peekingduck.declarative_loader.DeclarativeLoader>`
            to load the YAML file according to PeekingDuck's specified schema
            to obtain the declared nodes that would be sequentially initialized
            and used to create the pipeline for running inference.
        config_updates_cli (:obj:`str` | :obj:`None`): Configuration changes
            passed as part of the CLI command used to modify the node
            configurations directly from CLI.
        custom_nodes_parent_subdir (:obj:`str` | :obj:`None`): Relative path to
            a folder which contains custom nodes that users have created to be
            used with PeekingDuck. For more information on using custom nodes,
            please refer to
            `Getting Started <getting_started/03_custom_nodes.html>`_.
        nodes (:obj:`List[AbstractNode]` | :obj:`None`): If a list of nodes is
            provided, initialize by the node stack directly.
    """

    def __init__(
        self,
        run_config_path: Path = None,
        config_updates_cli: str = None,
        custom_nodes_parent_subdir: str = None,
        nodes: List[AbstractNode] = None,
    ):
        self.logger = logging.getLogger(__name__)
        try:
            if nodes:
                # instantiated_nodes is created differently when given nodes
                self.pipeline = Pipeline(nodes)
            elif run_config_path and config_updates_cli and custom_nodes_parent_subdir:
                # create Graph to run
                self.node_loader = DeclarativeLoader(
                    run_config_path, config_updates_cli, custom_nodes_parent_subdir
                )
                self.pipeline = self.node_loader.get_pipeline()
            else:
                raise ValueError(
                    "Arguments error! Pass in either nodes to load directly via "
                    "Pipeline or run_config_path, config_updates_cli, and "
                    "custom_nodes_parent_subdir to load via DeclarativeLoader."
                )
        except ValueError as error:
            self.logger.error(str(error))
            sys.exit(1)
        if RequirementChecker.n_update > 0:
            sys.exit(3)

    def run(self) -> None:
        """execute single or continuous inference"""
        while not self.pipeline.terminate:
            for node in self.pipeline.nodes:
                if (
                    "pipeline_end" in self.pipeline.data
                    and self.pipeline.data["pipeline_end"]
                ):
                    self.pipeline.terminate = True
                    if "pipeline_end" not in node.inputs:
                        continue

                if "all" in node.inputs:
                    inputs = copy.deepcopy(self.pipeline.data)
                else:
                    inputs = {
                        key: self.pipeline.data[key]
                        for key in node.inputs
                        if key in self.pipeline.data
                    }

                outputs = node.run(inputs)
                self.pipeline.data.update(outputs)
        # clean up nodes with threads
        for node in self.pipeline.nodes:
            if node.name.endswith(".live") or node.name.endswith(".recorded"):
                node.release_resources()

    def get_run_config(self) -> NodeList:
        """Retrieves run configuration.

        Returns:
            (:obj:`Dict`): Run configurations being used by runner.
        """
        return self.node_loader.node_list
