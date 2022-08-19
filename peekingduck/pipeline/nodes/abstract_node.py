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
Abstract Node class for all nodes.
"""

import collections
import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from typeguard import check_type

from peekingduck.config_loader import ConfigLoader
from peekingduck.utils.detect_id_mapper import obj_det_change_class_name_to_id


class AbstractNode(metaclass=ABCMeta):
    """Abstract Node class for inheritance by nodes.

    Defines default attributes and methods of a node.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
        node_path (:obj:`str`): Period-separated (``.``) relative path to the
            node from the ``peekingduck`` directory. **Default: ""**.
        pkd_base_dir (:obj:`pathlib.Path` | :obj:`None`): Path to
            ``peekingduck`` directory.
    """

    def __init__(
        self,
        config: Dict[str, Any] = None,
        node_path: str = "",
        pkd_base_dir: Optional[Path] = None,
        **kwargs: Any,
    ) -> None:
        self._name = node_path
        self.logger = logging.getLogger(self._name)

        if not pkd_base_dir:
            pkd_base_dir = Path(__file__).resolve().parents[2]

        self.node_name = ".".join(node_path.split(".")[-2:])

        # This is only initialized when the `optional_inputs` key is found in
        # the nodes' config file
        self.optional_inputs: List[str]

        # NOTE: ``config`` and ``kwargs_config`` are similar but are from
        # different inputs. ``config`` is when users input a dictionary to
        # update the node. ``kwargs_config`` is when users input parameters to
        # update the node
        self.config_loader = ConfigLoader(pkd_base_dir)
        self.load_node_config(config, kwargs)  # type: ignore

        # For object detection nodes, convert class names to class ids, if any
        if self.node_name in ["model.yolo", "model.efficientdet", "model.yolox"]:
            key = "detect" if hasattr(self, "detect") else "detect_ids"
            current_ids = self.config[key]
            _, updated_ids = obj_det_change_class_name_to_id(
                self.node_name, key, current_ids
            )
            # replace "detect_ids" with new "detect"
            self.config["detect"] = updated_ids

    @classmethod
    def __subclasshook__(cls: Any, subclass: Any) -> bool:
        return hasattr(subclass, "run") and callable(subclass.run)

    @abstractmethod
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """abstract method needed for running node"""
        raise NotImplementedError("This method needs to be implemented")

    # pylint: disable=R0201, W0107
    def release_resources(self) -> None:
        """To gracefully release any acquired system resources, e.g. webcam
        NOTE: To be overridden by subclass if required"""
        pass

    @property
    def inputs(self) -> List[str]:
        """Input requirements."""
        return self.input

    @property
    def outputs(self) -> List[str]:
        """Node outputs."""
        return self.output

    @property
    def name(self) -> str:
        """Node name."""
        return self._name

    def load_node_config(
        self, config: Dict[str, Any], kwargs_config: Dict[str, Any]
    ) -> None:
        """Loads node configuration.

        NOTE: ``config`` and ``kwargs_config`` are similar but come from
        different inputs. ``config`` is when users input a dictionary to update
        the node and ``kwargs_config`` is when users input parameters to update
        the node.

        Args:
            config (:obj:`Dict[str, Any]`): Loads configuration from a
                dictionary input.
            kwargs_config (:obj:`Dict[str, Any]`): Loads configuration from
                ``kwargs``.
        """
        # if full set of configuration is not included in config
        # load configuration and update node with **kwargs where possible
        # else load from kwargs only
        self.config = config
        if not self.config:
            loaded_config = self.config_loader.get(self.node_name)
            updated_config = self._edit_config(loaded_config, kwargs_config)
            self.config = updated_config

        self._check_type(self.config, self._get_config_types())
        # sets class attributes
        for key in self.config:
            setattr(self, key, self.config[key])

    def _check_type(
        self, config: Dict[str, Any], config_types: Dict[str, Any], parent: str = ""
    ) -> None:
        """Checks the typing of the config values against the provided
        `config_types`. Recursively checks each value of dictionaries.
        """
        for key in config:
            full_key = f"{parent}{key}"
            if isinstance(config[key], dict):
                self._check_type(config[key], config_types, f"{full_key}.")
            config_type = config_types.get(full_key, Any)
            check_type(f"{self.node_name}'s `{full_key}`", config[key], config_type)

    def _edit_config(
        self,
        dict_orig: Dict[str, Any],
        dict_update: Union[Dict[str, Any], collections.abc.Mapping],
    ) -> Dict[str, Any]:
        """Update value of a nested dictionary of varying depth using
        recursion
        """
        if dict_update:
            for key, value in dict_update.items():
                if isinstance(value, collections.abc.Mapping):
                    dict_orig[key] = self._edit_config(dict_orig.get(key, {}), value)
                elif key not in dict_orig:
                    self.logger.warning(
                        f"Config for node {self.node_name} does not have the key: {key}"
                    )
                else:
                    dict_orig[key] = value
                    self.logger.info(
                        f"Config for node {self.node_name} is updated to: '{key}': {value}"
                    )
        return dict_orig

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {}
