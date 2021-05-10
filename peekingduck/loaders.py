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

    def __init__(self) -> None:
        self._rootdir = os.path.join(
            os.path.dirname(os.path.abspath(__file__))
        )

    def _get_config_path(self, node: str) -> str:
        """ Based on the node, return the corresponding node config path """

        configs_folder = os.path.join(self._rootdir, 'configs')
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
        node_config['root'] = self._rootdir
        return node_config


class DeclarativeLoader:
    """Uses the declarative run_config.yml to load pipelines or compiled configs"""

    def __init__(self, node_configs: ConfigLoader,
                 run_config: str,
                 custom_folder_path: str = 'src/custom_nodes') -> None:
        self.logger = logging.getLogger(__name__)

        self.node_configs = node_configs
        self.node_list = self._load_node_list(run_config)
        self.custom_folder_path = custom_folder_path

    def _load_node_list(self, run_config: str) -> List[str]:
        """Loads a list of nodes from run_config.yml"""
        with open(run_config) as node_yml:
            nodes = yaml.load(node_yml, Loader=yaml.FullLoader)['nodes']

        self.logger.info(
            'Successfully loaded run_config file.')
        return nodes

    def compile_configrc(self) -> None:
        """Given a list of nodes, return compiled configs"""
        if os.path.isfile('node_config.yml'):
            os.remove('node_config.yml')
        with open('node_config.yml', 'a') as compiled_node_config:
            for node_str in self.node_list:
                node_type, node = node_str.split('.')
                if node_type == 'custom':
                    node_config_path = os.path.join(
                        self.custom_folder_path, node, 'config.yml')
                else:
                    dir_path = os.path.dirname(os.path.realpath(__file__))
                    config_filename = node + '.yml'
                    node_config_path = os.path.join(
                        dir_path, 'configs', node_type, config_filename)
                if os.path.isfile(node_config_path):
                    with open(node_config_path, 'r') as node_yml:
                        node_config = yaml.load(
                            node_yml, Loader=yaml.FullLoader)
                        node_config = {node_str: node_config}
                    yaml.dump(node_config, compiled_node_config,
                              default_flow_style=False)
                else:
                    self.logger.info(
                        "No associated configs found for %s. Skipping", node)

    def _import_nodes(self) -> List[Any]:
        """Given a list of nodes, import the appropriate nodes"""
        imported_nodes = []
        for node_str in self.node_list:
            node_type, node = node_str.split('.')
            if node_type == 'custom':
                custom_node_path = os.path.join(
                    self.custom_folder_path, node + '.py')
                spec = importlib.util.spec_from_file_location(  # type: ignore
                    node, custom_node_path)
                module = importlib.util.module_from_spec(spec)  # type: ignore
                spec.loader.exec_module(module)
                imported_nodes.append(("custom", module))
            else:
                imported_nodes.append((node_str, importlib.import_module(
                    'peekingduck.pipeline.nodes.' + node_str)))
            self.logger.info('%s added to pipeline.', node)
        return imported_nodes

    def _instantiate_nodes(self, imported_nodes: List[Any]) -> List[AbstractNode]:
        """ Given a list of imported nodes, instantiate nodes"""
        instantiated_nodes = []
        for node_name, node in imported_nodes:
            if node_name == 'custom':
                instantiated_nodes.append(node.Node(None))
            else:
                config = self.node_configs.get(node_name)
                instantiated_nodes.append(node.Node(config))
        return instantiated_nodes

    def get_nodes(self) -> Pipeline:
        """Returns a compiled Pipeline for PeekingDuck runner to execute"""
        imported_nodes = self._import_nodes()
        instantiated_nodes = self._instantiate_nodes(imported_nodes)

        try:
            return Pipeline(instantiated_nodes)
        except ValueError as error:
            self.logger.error(str(error))
            sys.exit(1)
