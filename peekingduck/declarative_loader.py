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
Helper classes to load configs, nodes
"""

import sys
import os
import importlib
import logging
import ast
import collections.abc
import re
from typing import Any, Dict, List

import yaml

from peekingduck.pipeline.pipeline import Pipeline
from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.configloader import ConfigLoader

PEEKINGDUCK_NODE_TYPE = ["input", "model", "draw", "dabble", "output"]


class DeclarativeLoader:  # pylint: disable=too-few-public-methods
    """
    A helper class to create pipeline.

    The declarative loader class creates the specified nodes according to any
    modfications provided in the configs and returns the pipeline needed for
    inference.

    Args:

        run_config (:obj:`str`): Path to yaml file that declares the node \
        sequence to be used in the pipeline

        config_updates_cli (:obj:`str`): stringified nested dictionaries of \
        config changes passed as part of cli command. Used to modify the node \
        configurations direct from cli.

        custom_node_parent_folder (:obj:`str`): path to parent folder which contains \
        custom nodes that users have created to be used with PeekingDuck. \
        For more information on using custom nodes, please refer to \
        `getting started <getting_started/03_custom_nodes.html>`_.

    """

    def __init__(self,
                 run_config: str,
                 config_updates_cli: str,
                 custom_node_parent_folder: str) -> None:
        self.logger = logging.getLogger(__name__)

        pkdbasedir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        self.config_loader = ConfigLoader(pkdbasedir)

        self.node_list = self._load_node_list(run_config)
        self.config_updates_cli = ast.literal_eval(
            config_updates_cli)  # type: ignore

        custom_folder = self._get_custom_name_from_node_list()
        if custom_folder is not None:
            custom_folder_path = os.path.join(os.getcwd(),
                                              custom_node_parent_folder,
                                              custom_folder)
            self.custom_config_loader = ConfigLoader(custom_folder_path)
            sys.path.append(custom_node_parent_folder)

            self.custom_folder_path = custom_folder_path

    def _load_node_list(self, run_config: str) -> List[str]:
        """Loads a list of nodes from run_config.yml"""
        with open(run_config) as node_yml:
            nodes = yaml.safe_load(node_yml)['nodes']

        self.logger.info('Successfully loaded run_config file.')
        return nodes

    def _get_custom_name_from_node_list(self) -> Any:

        custom_name = None

        for node in self.node_list:
            if isinstance(node, dict) is True:
                node_type = [*node][0].split(".")[0]
            else:
                node_type = node.split(".")[0]

            if node_type not in PEEKINGDUCK_NODE_TYPE:
                custom_name = node_type
                break

        return custom_name

    def _instantiate_nodes(self) -> List[AbstractNode]:
        """ Given a list of imported nodes, instantiate nodes"""
        instantiated_nodes = []

        for node_item in self.node_list:
            config_updates_yml = None
            node_str = node_item

            if isinstance(node_item, dict):
                node_str = list(node_item.keys())[0]  # type: ignore
                config_updates_yml = node_item[node_str]

            node_str_split = node_str.split('.')

            msg = "Initialising " + node_str + " node..."
            self.logger.info(msg)

            if len(node_str_split) == 3:
                # convert windows/linux filepath to a module path
                path_to_node = ".".join(re.split(r'\\|\/', self.custom_folder_path)[-1:]) + "."
                node_name = ".".join(node_str_split[-2:])

                instantiated_node = self._init_node(path_to_node,
                                                    node_name,
                                                    self.custom_config_loader,
                                                    config_updates_yml)  # type: ignore
            else:
                path_to_node = 'peekingduck.pipeline.nodes.'

                instantiated_node = self._init_node(path_to_node,
                                                    node_str,
                                                    self.config_loader,
                                                    config_updates_yml)  # type: ignore

            instantiated_nodes.append(instantiated_node)

        return instantiated_nodes

    def _init_node(self, path_to_node: str, node_name: str,
                   config_loader: ConfigLoader,
                   config_updates_yml: Dict[str, Any]) -> AbstractNode:
        """ Import node to filepath and initialise node with config """

        node = importlib.import_module(path_to_node + node_name)
        config = config_loader.get(node_name)

        # First, override default configs with values from run_config.yml
        if config_updates_yml is not None:
            config = self._edit_config(config, config_updates_yml, node_name)

        # Second, override configs again with values from cli
        if self.config_updates_cli is not None:
            if node_name in self.config_updates_cli.keys():
                config = self._edit_config(
                    config, self.config_updates_cli[node_name], node_name)

        return node.Node(config)  # type: ignore

    def _edit_config(self, dict_orig: Dict[str, Any],
                     dict_update: Dict[str, Any],
                     node_name: str) -> Dict[str, Any]:
        """ Update value of a nested dictionary of varying depth using
            recursion
        """
        for key, value in dict_update.items():
            if isinstance(value, collections.abc.Mapping):
                dict_orig[key] = self._edit_config(
                    dict_orig.get(key, {}), value, node_name)  # type: ignore
            else:
                if key not in dict_orig:
                    self.logger.warning(
                        "Config for node %s does not have the key: %s",
                        node_name, key)
                else:
                    dict_orig[key] = value
                    self.logger.info(
                        "Config for node %s is updated to: '%s': %s",
                        node_name, key, value)
        return dict_orig

    def get_pipeline(self) -> Pipeline:
        """Returns a compiled Pipeline for PeekingDuck runner to execute"""
        instantiated_nodes = self._instantiate_nodes()

        try:
            return Pipeline(instantiated_nodes)
        except ValueError as error:
            self.logger.error(str(error))
            sys.exit(1)
