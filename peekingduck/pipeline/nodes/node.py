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

import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List


class AbstractNode(metaclass=ABCMeta):
    """Abstract Node class that defines requirements for a node."""

    def __init__(self, config: Dict[str, Any], node_path: str) -> None:

        self._inputs = config['input']
        self._outputs = config['output']
        self._name = node_path

        self.logger = logging.getLogger(self._name)

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
        return self._inputs

    @property
    def outputs(self) -> List[str]:
        """getter for node outputs"""
        return self._outputs

    @property
    def name(self) -> str:
        """getter for node name"""
        return self._name
