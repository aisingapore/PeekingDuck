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

import os
import sys
import importlib
import logging
from typing import List, Dict, Any
import yaml
from peekingduck.pipeline.pipeline import Pipeline
from peekingduck.config import ConfigLoader
from peekingduck.pipeline.nodes.node import AbstractNode

END_TYPE = 'process_end'

"""
Combine runner at this level. Use this to create the graph, and waddle is loop or once
"""

class Runner():
    """Runner class that uses the declared nodes to create pipeline to run inference
    """

    def __init__(self, RUN_PATH: str, CUSTOM_NODE_PATH: str = None,
                 nodes: List[AbstractNode] = None):
        """
        Args:
            RUN_PATH (str): path to yaml file of node pipeine declaration.
            CUSTOM_NODE_PATH (str): path to custom nodes folder if used.
            nodes (:obj:'list' of :obj:'Node'): if not using declarations via yaml,
                initialize by giving the node stack as a list

        """

        self.logger = logging.getLogger(__name__)

        instantiated_nodes = []

        if not nodes:
            with open(RUN_PATH) as file:
                self.run_config = yaml.load(file, Loader=yaml.FullLoader)
                self.logger.info(
                    'Successfully loaded run_config file. Proceeding to create Graph.')
            # create Graph to run
            nodes_config = ConfigLoader(self.run_config['nodes'])
            imported_nodes = []
            for node in self.run_config['nodes']:
                if node.split('.')[0] == 'custom':
                    custom_node_path = os.path.join(
                        CUSTOM_NODE_PATH, node.split('.')[1]+'.py')
                    spec = importlib.util.spec_from_file_location(
                        node, custom_node_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    imported_nodes.append(("custom", module))
                else:
                    imported_nodes.append((node, importlib.import_module(
                        'peekingduck.pipeline.nodes.' + node)))
                self.logger.info("%s added to pipeline.", node)

            # instantiate classes from imported nodes
            for node_name, node in imported_nodes:
                if node_name == 'custom':
                    instantiated_nodes.append(node.Node(None))
                else:
                    config = nodes_config.get(node_name)
                    instantiated_nodes.append(node.Node(config))

        # If Runner given nodes, instantiated_nodes is created differently
        else:
            for node in nodes:
                instantiated_nodes.append(node)

        # Create Graph
        try:
            self.pipeline = Pipeline(instantiated_nodes)
        except ValueError as exception:
            self.logger.error(str(exception))
            sys.exit(1)

    def run(self) -> None:
        """execute single or continuous inference
        """
        while not self.pipeline.video_end:
            self.pipeline.execute()

    def get_run_config(self) -> Dict[str, Any]:
        """retreive run configs

        Returns:
            Dict[Any]: run configs being used for runner
        """
        return self.run_config
