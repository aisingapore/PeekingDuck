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

import sys
import os
import importlib
import logging
from typing import Any, Dict, List

import yaml

from peekingduck.pipeline.pipeline import Pipeline
from peekingduck.pipeline.nodes.node import AbstractNode


class ConfigLoader:  # pylint: disable=too-few-public-methods
    """ Reads configuration and returns configuration required to
    create the nodes for the project
    """

    def __init__(self, basedir: str) -> None:
        self._basedir = basedir

    def _get_config_path(self, node: str) -> str:
        """ Based on the node, return the corresponding node config path """

        configs_folder = os.path.join(self._basedir, 'configs')
        node_type, node_name = node.split(".")
        node_name = node_name + ".yml"
        filepath = os.path.join(configs_folder, node_type, node_name)

        return filepath

    def get(self, node_name: str) -> Dict[str, Any]:
        """Get item from configuration read from the filepath,
        item refers to the node item configuration you are trying to get"""

        filepath = self._get_config_path(node_name)

        with open(filepath) as file:
            node_config = yaml.load(file, Loader=yaml.FullLoader)

        # some models require the knowledge of where the root is for loading
        node_config['root'] = self._basedir
        return node_config


class DeclarativeLoader:  # pylint: disable=too-few-public-methods
    """Uses the declarative run_config.yml to load pipelines or compiled configs"""

    def __init__(self,
                 run_config: str,
                 custom_folder_path: str = 'src/custom_nodes') -> None:
        self.logger = logging.getLogger(__name__)

        pkdbasedir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

        self.config_loader = ConfigLoader(pkdbasedir)
        self.custom_config_loader = ConfigLoader(custom_folder_path)
        self.node_list = self._load_node_list(run_config)
        self.custom_folder_path = custom_folder_path

        # add custom nodes path "src" for module discovery
        sys.path.append(custom_folder_path.split("/")[-2])

    def _load_node_list(self, run_config: str) -> List[str]:
        """Loads a list of nodes from run_config.yml"""
        with open(run_config) as node_yml:
            nodes = yaml.load(node_yml, Loader=yaml.FullLoader)['nodes']

        self.logger.info('Successfully loaded run_config file.')
        return nodes

    def _instantiate_nodes(self) -> List[AbstractNode]:
        """ Given a list of imported nodes, instantiate nodes"""
        instantiated_nodes = []

        for node_item in self.node_list:
            config_updates = None
            node_str = node_item

            if isinstance(node_item, dict):
                node_str = list(node_item.keys())[0]  # type: ignore
                config_updates = {key: d[key] for d in node_item[node_str] for key in d}

            node_str_split = node_str.split('.')

            msg = "Initialising " + node_str + " node..."
            self.logger.info(msg)

            if len(node_str_split) == 3:
                path_to_node = ".".join(self.custom_folder_path.split('/')[-1:]) + "."
                node_name = ".".join(node_str_split[-2:])

                instantiated_node = self._init_node(path_to_node,
                                                    node_name,
                                                    self.custom_config_loader,
                                                    config_updates)  # type: ignore
            else:
                path_to_node = 'peekingduck.pipeline.nodes.'

                instantiated_node = self._init_node(path_to_node,
                                                    node_str,
                                                    self.config_loader,
                                                    config_updates)  # type: ignore

            instantiated_nodes.append(instantiated_node)

        return instantiated_nodes

    def _init_node(self, path_to_node: str, node_name: str,
                   config_loader: ConfigLoader,
                   config_updates: Dict[str, Any]) -> AbstractNode:
        """ Import node to filepath and initialise node with config """

        node = importlib.import_module(path_to_node + node_name)
        config = config_loader.get(node_name)

        if config_updates is not None:
            config = self._edit_node_config(config, config_updates)

        return node.Node(config)  # type: ignore

    def _edit_node_config(self, config: Dict[str, Any],
                          config_updates: Dict[str, Any]) -> Dict[str, Any]:
        " Edit default node configuration"

        params = set(config_updates.keys()) - set(config.keys())
        for param in params:
            config_updates.pop(param)
            msg = "'" + param + "' is not a valid configurable parameter"
            self.logger.warning(msg)
        config.update(config_updates)

        return config

    def get_nodes(self) -> Pipeline:
        """Returns a compiled Pipeline for PeekingDuck runner to execute"""
        instantiated_nodes = self._instantiate_nodes()

        try:
            return Pipeline(instantiated_nodes)
        except ValueError as error:
            self.logger.error(str(error))
            sys.exit(1)
