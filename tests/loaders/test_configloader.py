# Copyright 2021 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from peekingduck.loaders import ConfigLoader
import pytest
from typing import Dict
import os
import tempfile
import yaml


RUN_PATH = os.path.dirname(os.path.realpath(__file__))
CONFIG_FOLDER_PATH = os.path.join(RUN_PATH, "configs")


def create_configloader():
    config_loader = ConfigLoader(RUN_PATH)

    return config_loader


def create_config_yaml(data):
    temp_dir = tempfile.TemporaryDirectory(dir=CONFIG_FOLDER_PATH)
    temp_file = tempfile.NamedTemporaryFile(dir=temp_dir.name,
                                            suffix=".yml",
                                            delete=False)
    with open(temp_file.name, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    return temp_file.name


class TestConfigLoader():

    def test_config_loader_returns_correct_config_filepath(self):

        node = 'type.node'
        config_loader = create_configloader()
        filepath = config_loader._get_config_path(node)

        ground_truth = os.path.join(CONFIG_FOLDER_PATH,
                                    node.replace(".", "/"))
        ground_truth = ground_truth + ".yml"

        assert filepath == ground_truth

    def test_config_loader_load_correct_node_config(self):
        data = {"input": "img",
                "output": "img"}

        yaml_file_path = create_config_yaml(data)

        yaml_file_path = yaml_file_path.replace("/", ".")
        yaml_file_path = yaml_file_path.split(".")
        node = yaml_file_path[-3] + "." + yaml_file_path[-2]

        config_loader = create_configloader()
        node_config = config_loader.get(node)

        assert yaml_file_path == 'test'
