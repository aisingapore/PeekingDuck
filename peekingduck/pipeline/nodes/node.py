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
Abstract Node class for all nodes
"""

import pathlib
import logging
import collections
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List

from peekingduck.configloader import ConfigLoader


class AbstractNode(metaclass=ABCMeta):
    """
    Abstract Node class for inheritance by nodes.
    It defines default attributes and methods of a node.
    """

    def __init__(self, config: Dict[str, Any] = None,
                 node_path: str = "", pkdbasedir: str = None, **kwargs: Any) -> None:

        self._name = node_path
        self.logger = logging.getLogger(self._name)

        if not pkdbasedir:
            pkdbasedir = str(pathlib.Path(__file__).parents[2].resolve())

        self.node_name = ".".join(node_path.split(".")[-2:])

        # NOTE: config and kwargs_config are similar but are from different inputs
        # config is when users input a dictionary to update the node
        # kwargs_config is when users input parameters to update the node
        self.config_loader = ConfigLoader(pkdbasedir)
        self.load_node_config(config, kwargs)  # type: ignore

    @classmethod
    def __subclasshook__(cls: Any, subclass: Any) -> bool:
        return (hasattr(subclass, 'run') and
                callable(subclass.run))

    @abstractmethod
    def run(self, inputs: Dict[str, Any]) -> None:
        """abstract method needed for running node"""
        raise NotImplementedError("This method needs to be implemented")

    @property
    def inputs(self) -> List[str]:
        """getter for input requirements"""
        return self.input

    @property
    def outputs(self) -> List[str]:
        """getter for node outputs"""
        return self.output

    @property
    def name(self) -> str:
        """getter for node name"""
        return self._name

    def load_node_config(self,
                         config: Dict[str, Any],
                         kwargs_config: Dict[str, Any]) -> None:
        """ loads node configuration
        NOTE: config and kwargs_config are similar but come from different inputs
        config is when users input a dictionary to update the node
        kwargs_config is when users input parameters to update the node

        Args:
            config (Dict[str, Any]): loads configuration from a dictionary input
            kwargs_config (Dict[str, Any]): loads configuration from kwargs
        """

        # if full set of configuration is not included in config
        # load configuration and update node with **kwargs where possible
        # else load from kwargs only
        self.config = config
        if not self.config:
            loaded_config = self.config_loader.get(self.node_name)
            updated_config = self._edit_config(loaded_config, kwargs_config)
            self.config = updated_config

        # sets class attributes
        for key in self.config:
            setattr(self, key, self.config[key])

    # pylint: disable=R0801
    def _edit_config(self,
                     dict_orig: Dict[str, Any],
                     dict_update: Dict[str, Any]) -> Dict[str, Any]:
        """ Update value of a nested dictionary of varying depth using
            recursion
        """
        if dict_update:
            for key, value in dict_update.items():
                if isinstance(value, collections.abc.Mapping):
                    dict_orig[key] = self._edit_config(
                        dict_orig.get(key, {}), value)  # type: ignore
                elif key not in dict_orig:
                    self.logger.warning(
                        "Config for node %s does not have the key: %s",
                        self.node_name, key)
                else:
                    dict_orig[key] = value
                    self.logger.info(
                        "Config for node %s is updated to: '%s': %s",
                        self.node_name, key, value)
        return dict_orig
