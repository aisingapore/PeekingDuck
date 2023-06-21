# Copyright 2022 AI Singapore
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
Helper classes to load configurations and nodes.
"""

import ast
import collections.abc
import importlib
import locale
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

import yaml

from peekingduck.config_loader import ConfigLoader
from peekingduck.nodes.abstract_node import AbstractNode
from peekingduck.pipeline import Pipeline
from peekingduck.utils.detect_id_mapper import obj_det_change_class_name_to_id

PEEKINGDUCK_NODE_TYPES = ["input", "augment", "model", "draw", "dabble", "output"]


class DeclarativeLoader:  # pylint: disable=too-few-public-methods, too-many-instance-attributes
    """A helper class to create
    :py:class:`Pipeline <peekingduck.pipeline.Pipeline>`.

    The declarative loader class creates the specified nodes according to any
    modifications provided in the configs and returns the pipeline needed for
    inference.

    Args:
        pipeline_path (:obj:`pathlib.Path`): Path to a YAML file that
            declares the node sequence to be used in the pipeline.
        config_updates_cli (:obj:`str`): Stringified nested dictionaries of
            configuration changes passed as part of CLI command. Used to modify
            the node configurations directly from the CLI.
        custom_nodes_parent_subdir (:obj:`str`): Relative path to parent
            folder which contains custom nodes that users have created to be
            used with PeekingDuck. For more information on using custom nodes,
            please refer to
            `Getting Started <getting_started/03_custom_nodes.html>`_.
    """

    def __init__(
        self,
        pipeline_path: Path,
        config_updates_cli: str,
        custom_nodes_parent_subdir: str,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        self.pkd_base_dir = Path(__file__).resolve().parent
        self.config_loader = ConfigLoader(self.pkd_base_dir)

        if pipeline_path.parent != Path.cwd() and pipeline_path.is_absolute():
            parent_path = str(pipeline_path.parent)
            self.logger.info(f"change working directory to {parent_path}")
            os.chdir(parent_path)

        self.node_list = self._load_node_list(pipeline_path)
        self.config_updates_cli = ast.literal_eval(config_updates_cli)

        custom_nodes_name = self._get_custom_name_from_node_list()
        if custom_nodes_name is not None:
            custom_nodes_dir = (
                Path.cwd() / custom_nodes_parent_subdir / custom_nodes_name
            )
            self.custom_config_loader = ConfigLoader(custom_nodes_dir)
            sys.path.append(custom_nodes_parent_subdir)

            self.custom_nodes_dir = custom_nodes_dir

    def get_pipeline(self) -> Pipeline:
        """Returns a compiled `Pipeline` for PeekingDuck `Runner` to execute."""
        instantiated_nodes = self._instantiate_nodes()

        try:
            return Pipeline(instantiated_nodes)
        except ValueError as error:
            self.logger.error(str(error))
            sys.exit(1)

    def _edit_config(
        self, dict_orig: Dict[str, Any], dict_update: Dict[str, Any], node_name: str
    ) -> Dict[str, Any]:
        """Update value of a nested dictionary of varying depth using recursion"""
        for key, value in dict_update.items():
            if key not in dict_orig:
                self.logger.warning(
                    f"Config for node {node_name} does not have the key: {key}"
                )
                continue
            if isinstance(value, collections.abc.Mapping) and key != "callbacks":
                dict_orig[key] = self._edit_config(
                    dict_orig.get(key, {}), value, node_name  # type: ignore
                )
                continue
            # `value` below will not be a Dict since the keys are not `callbacks`
            if key == "detect":
                value = self._change_class_name_to_id(node_name, cast(List[Any], value))

            dict_orig[key] = value
            self.logger.info(
                f"Config for node {node_name} is updated to: '{key}': {value}"
            )
        return dict_orig

    def _get_custom_name_from_node_list(self) -> Any:
        custom_name = None

        for node_str, _ in self.node_list:
            node_type = node_str.split(".")[0]

            if node_type not in PEEKINGDUCK_NODE_TYPES:
                custom_name = node_type
                break

        return custom_name

    def _init_node(
        self,
        path_to_node: str,
        node_name: str,
        config_loader: ConfigLoader,
        config_updates_yml: Optional[Dict[str, Any]],
    ) -> AbstractNode:
        """Imports node to filepath and initializes node with config."""
        node = importlib.import_module(path_to_node + node_name)
        config = config_loader.get(node_name)

        # First, override default configs with values from pipeline_config.yml
        if config_updates_yml is not None:
            config = self._edit_config(config, config_updates_yml, node_name)

        # Second, override configs again with values from cli
        if (
            self.config_updates_cli is not None
            and node_name in self.config_updates_cli.keys()
        ):
            config = self._edit_config(
                config, self.config_updates_cli[node_name], node_name
            )
            if node_name in self.config_updates_cli.keys():
                config = self._edit_config(
                    config, self.config_updates_cli[node_name], node_name
                )

        return node.Node(config)

    def _instantiate_nodes(self) -> List[AbstractNode]:
        """Given a list of imported nodes, instantiate nodes"""
        instantiated_nodes = []

        for node_str, config_updates_yml in self.node_list:
            node_str_split = node_str.split(".")

            self.logger.info(f"Initializing {node_str} node...")

            if len(node_str_split) == 3:
                # convert windows/linux filepath to a module path
                path_to_node = f"{self.custom_nodes_dir.name}."
                node_name = ".".join(node_str_split[-2:])

                instantiated_node = self._init_node(
                    path_to_node,
                    node_name,
                    self.custom_config_loader,
                    config_updates_yml,
                )
            else:
                path_to_node = "peekingduck.nodes."

                instantiated_node = self._init_node(
                    path_to_node, node_str, self.config_loader, config_updates_yml
                )

            instantiated_nodes.append(instantiated_node)

        return instantiated_nodes

    def _load_node_list(self, pipeline_path: Path) -> "NodeList":
        """Loads a list of nodes from pipeline_path.yml"""
        with open(
            pipeline_path, encoding=locale.getpreferredencoding(False)
        ) as node_yml:
            data = yaml.safe_load(node_yml)
        if not isinstance(data, dict) or "nodes" not in data:
            raise ValueError(
                f"{pipeline_path} has an invalid structure. "
                "Missing top-level 'nodes' key."
            )

        nodes = data["nodes"]
        if nodes is None:
            raise ValueError(f"{pipeline_path} does not contain any nodes!")

        self.logger.info("Successfully loaded pipeline file.")
        return NodeList(nodes)

    @staticmethod
    def _change_class_name_to_id(node_name: str, value: List[Any]) -> List[int]:
        """Converts class names to detect IDs, skips yolo_face which has no
        class names and huggingface_hub which does not conform to master_map.yml.
        """
        if node_name not in {"model.yolo_face", "model.huggingface_hub"}:
            value = obj_det_change_class_name_to_id(node_name, value)
        return value


class NodeList:
    """Iterator class to return node string and node configs (if any) from the
    nodes declared in the run config file.
    """

    def __init__(self, nodes: List[Union[Dict[str, Any], str]]) -> None:
        self.nodes = nodes
        self.length = len(nodes)

    def __iter__(self) -> Iterator[Tuple[str, Optional[Dict[str, Any]]]]:
        self.current = -1
        return self

    def __next__(self) -> Tuple[str, Optional[Dict[str, Any]]]:
        self.current += 1
        if self.current >= self.length:
            raise StopIteration
        node_item = self.nodes[self.current]

        if isinstance(node_item, dict):
            node_str = next(iter(node_item))
            config_updates = node_item[node_str]
        else:
            node_str = node_item
            config_updates = None

        return node_str, config_updates
