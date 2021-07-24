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

# TODO move this disable to pylintrc file
# pylint: disable=E1101
class AbstractNode(metaclass=ABCMeta):
    """
    Abstract Node class for inheritance by nodes.
    It defines default attributes and methods of a node.
    """

    def __init__(self, config: Dict[str, Any], node_path: str) -> None:

        self._name = node_path
        self.logger = logging.getLogger(self._name)

        # load configuration
        pkdbasedir = pathlib.Path(__file__).parents[2].resolve()
        node_name = ".".join(node_path.split(".")[-2:])
        self.config_loader = ConfigLoader(pkdbasedir)
        loaded_config = self.config_loader.get(node_name)

        updated_config = self._edit_config(loaded_config, config, node_name)

        # sets class attributes
        for key in updated_config:
            setattr(self, key, updated_config[key])


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

    # TODO this updates the dictionary
    # pylint: disable=R0801
    def _edit_config(self,
                     dict_orig: Dict[str, Any],
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
