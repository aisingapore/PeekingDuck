""" Reads and manipulations configuration files
"""

import os
import yaml
import importlib
import logging
from typing import Any, Dict, List
from peekingduck.pipeline.pipeline import Pipeline
from peekingduck.pipeline.nodes.node import AbstractNode


class ConfigLoader:
    """ Reads configuration and returns configuration required to
    create the nodes for the project
    """

    def __init__(self):
        self._master_node_config = {}
        self._rootdir = os.path.join(
            os.path.dirname(os.path.abspath(__file__))
        )

        self._load_configs()

    def _load_configs(self):
        configs_folder = os.path.join(self._rootdir, 'configs')
        node_filepaths = [os.path.join(node_type_path, node)
                          for _, node_type_path, nodes in os.walk(configs_folder)
                          for node in nodes if node.endswith('.yml')]
        for filepath in node_filepaths:
            filepath_split = filepath.split('/')
            node_type, node = filepath_split[-2], filepath_split[-1][:-4]
            node_name = node_type + '.' + node
            with open(filepath) as file:
                node_config = yaml.load(file, Loader=yaml.FullLoader)
                self._master_node_config[node_name] = node_config

    def get(self, item: str) -> Dict[str, Any]:
        """Get item from configuration read from the filepath,
        item refers to the node item configuration you are trying to get"""

        node = self._master_node_config[item]

        # some models require the knowledge of where the root is for loading
        node['root'] = self._rootdir
        return node


class DeclarativeLoader:

    def __init__(self, node_configs: ConfigLoader, run_config: str, custom_folder_path: str):
        self.node_configs = node_configs
        self.node_list = self._load_node_list(run_config)
        self.custom_folder_path = custom_folder_path

        self.logger = logging.getLogger(__name__)

    def _load_node_list(self, run_config: str):
        """Loads a list of nodes from run_config.yml"""
        with open(run_config) as node_yml:
            nodes = yaml.load(node_yml, Loader=yaml.FullLoader)['nodes']
        return nodes

    def compile_configrc(self):
        if os.path.isfile('node_config.yml'):
            os.remove('node_config.yml')
        with open('node_config.yml', 'a') as compiled_node_config:
            for node_str in self.node_list:
                node_type, node = node_str.split('.')
                if node_type == 'custom':
                    node_config_path = os.path.join('src/custom_nodes', node, 'config.yml')
                else:
                    dir_path = os.path.dirname(os.path.realpath(__file__))
                    config_filename = node + '.yml'
                    node_config_path = os.path.join(dir_path, 'configs', node_type, config_filename)
                if os.path.isfile(node_config_path):
                    with open(node_config_path, 'r') as node_yml:
                        node_config = yaml.load(node_yml, Loader=yaml.FullLoader)
                        node_config = {node_str: node_config}
                    yaml.dump(node_config, compiled_node_config, default_flow_style=False)
                else:
                    self.logger(f'No associated configs found for {node}. Skipping')

    def _import_nodes(self):
        imported_nodes = []
        for node_str in self.node_list:
            node_type, node = node_str.split('.')
            if node_type == 'custom':
                custom_node_path = os.path.join(
                    self.custom_folder_path, node + '.py')
                spec = importlib.util.spec_from_file_location(
                    node, custom_node_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                imported_nodes.append(("custom", module))
            else:
                imported_nodes.append((node_str, importlib.import_module(
                    'peekingduck.pipeline.nodes.' + node)))
            logging.info("{} added to pipeline.".format(node))
        return imported_nodes

    def _instantiate_nodes(self, imported_nodes):
        instantiated_nodes = []
        for node_name, node in imported_nodes:
            if node_name == 'custom':
                instantiated_nodes.append(node.Node(None))
            else:
                config = self.node_configs.get(node_name)
                instantiated_nodes.append(node.Node(config))
        return instantiated_nodes

    def get_nodes(self):
        imported_nodes = self._import_nodes()
        instantiated_nodes = self._instantiate_nodes(imported_nodes)

        return Pipeline(instantiated_nodes)
