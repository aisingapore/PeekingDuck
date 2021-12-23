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
Loads configurations for individual nodes.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import logging
import yaml

# Master map file for class name to object IDs for object detection models
MASTER_MAP = "pipeline/nodes/model/master_map.yml"


class ConfigLoader:  # pylint: disable=too-few-public-methods
    """A helper class to create pipeline.

    The config loader class is used to allow for instantiation of Node classes
    directly instead of reading configurations from the run config yaml.

    Args:
        base_dir (:obj:`pathlib.Path`): Base directory of ``peekingduck``
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self.logger = logging.getLogger(__name__)

    def _get_config_path(self, node: str) -> Path:
        """Based on the node, return the corresponding node config path"""
        configs_folder = self._base_dir / "configs"
        node_type, node_name = node.split(".")
        file_path = configs_folder / node_type / f"{node_name}.yml"

        return file_path

    @staticmethod
    def _load_mapping(node_name: str) -> Dict[str, int]:
        """Loads class name to object ID mapping from the file
        peekingduck/pipeline/nodes/model/master_map.yml

        Tech Notes
            master_map.yml comprises two documents:
            1. model mapping: tells which mapping system a particular object detection model uses
            2. class name to ID mapping: maps class name to ID, supports multiple mapping systems

        Args:
            node_name (str): Tells function which mapping to load,
                             Possible values = { model.efficientdet, model.yolo, model.yolox }.

        Returns:
            Dict[str, int]: Mapping of class names to object IDs relevant to given node_name
        """
        # use __file__ instead of self._base_dir as latter can be set to any (temp) path
        # without `peekingduck/` subdirectory, resulting in master map file not found error
        master_map_file = Path(__file__).parent / MASTER_MAP
        # read both documents from master_map.yml
        with master_map_file.open() as map_file:
            model_map_dict, class_id_map_dict = yaml.safe_load_all(map_file)

        # node_name sanity check, to preempt non-object detection model nodes
        node_name_list = list(map(lambda x: f"model.{x}", model_map_dict.keys()))
        assert (
            node_name in node_name_list
        ), f"Node Name Error: expect one of {node_name_list} but got {node_name}"

        # determine mapping system to use based on given model
        model = node_name.replace("model.", "")
        model_map = model_map_dict[model]

        # construct class name->object ID map for the model
        class_id_map = {}
        for key, val in class_id_map_dict.items():
            class_id = val[model_map]
            class_id_map[key.lower()] = class_id
        return class_id_map

    def get(self, node_name: str) -> Dict[str, Any]:
        """Gets node configuration for specified node.

        Args:
            node_name (:obj:`str`): Name of node.

        Returns:
            node_config (:obj:`Dict[str, Any]`): A dictionary of node
            configurations for the specified node.
        """
        file_path = self._get_config_path(node_name)

        with open(file_path) as file:
            node_config = yaml.safe_load(file)

        # some models require the knowledge of where the root is for loading
        node_config["root"] = self._base_dir
        return node_config

    def change_class_name_to_id(
        self, node_name: str, key: str, value: List[Any]
    ) -> Tuple[str, List[int]]:
        """Process object detection model node's detect_ids key and check for
        any class names to be converted to object IDs.
        E.g. person to 0, car to 2

        Args:
            node_name (str): to determine which object detection model is being used
                             because different models can use different object IDs.
            key (str): expected to be "detect_ids"; error otherwise.
            value (List[Any]): list of class names or object IDs for detection.
                               If object IDs, do nothing.
                               If class names, convert to object IDs.

        Returns:
            Tuple[str, List[int]]: "detect_ids", list of sorted object IDs.
        """
        class_id_map = self._load_mapping(node_name)

        if not value:
            self.logger.warning("detect_ids list is empty, defaulting to detect person")
            value = ["person"]
        elif value == ["*"]:
            self.logger.info("Detecting all object classes")
            value = [*class_id_map]

        value_lc = [x.lower() if isinstance(x, str) else x for x in value]

        # parse value_lc for possible class name errors
        invalid_class_names = []
        for class_name in value_lc:
            if isinstance(class_name, str) and class_id_map.get(class_name, -1) < 0:
                invalid_class_names.append(class_name)

        if invalid_class_names:
            self.logger.warning(f"Invalid class names: {invalid_class_names}")

        # convert class names to numeric object IDs, any errors default to zero
        obj_ids_set = {
            x if isinstance(x, int) else class_id_map.get(x, 0) for x in value_lc
        }
        obj_ids_sorted_list = sorted(list(obj_ids_set))
        return key, obj_ids_sorted_list
