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
from typing import Any, Dict, List
import yaml
from peekingduck.pipeline.nodes.node import AbstractNode

class ConfigLoader:
    """ Reads configuration and returns configuration required to
    create the nodes for the project
    """
    def __init__(self, nodes: List[AbstractNode], node_yml: str = None):
        self._master_node_config = {}
        self._rootdir = os.path.join(
            os.path.dirname(os.path.abspath(__file__))
        )
        if node_yml:
            node_config = self._load_from_path(node_yml)
            self._master_node_config = node_config
        else:
            self.load_from_node_list(nodes)

    def load_from_node_list(self, nodes: List[AbstractNode]) -> None:
        """load node_configs from a list of nodes.
        Configs is expected to be at level of peekingduck"""
        for node in nodes:
            node_type, node_name = node.split('.')
            config_filename = node_name + '.yml'
            filepath = os.path.join(self._rootdir, 'configs', node_type, config_filename)
            node_config = self._load_from_path(filepath)
            self._master_node_config[node] = node_config

    @staticmethod
    def _load_from_path(filepath: str) -> None:
        """load node_configs directly from a custom node_config"""
        with open(filepath) as file:
            node_config = yaml.load(file, Loader=yaml.FullLoader)
        return node_config

    def get(self, item: str) -> Dict[str, Any]:
        """Get item from configuration read from the filepath,
        item refers to the node item configuration you are trying to get"""

        node = self._master_node_config[item]

        # some models require the knowledge of where the root is for loading
        node['root'] = self._rootdir
        return node
