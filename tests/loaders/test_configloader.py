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

import os
import yaml

import pytest
from peekingduck.declarative_loader import ConfigLoader


@pytest.fixture
def configloader():
    config_loader = ConfigLoader("tmp_dir")

    return config_loader


def create_config_yaml(node, data):
    node_type, node_name = node.split(".")
    config_path = os.path.join("tmp_dir", 'configs')

    node_config_path = os.path.join(config_path, node_type)
    os.makedirs(node_config_path)
    config_file = node_name + ".yml"

    full_path = os.path.join(node_config_path, config_file)
    with open(full_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


@pytest.mark.usefixtures("tmp_dir")
class TestConfigLoader:

    def test_config_loader_returns_correct_config_filepath(self, configloader):

        node = 'type.node'
        # .replace("\\","/") for windows where os.path.join uses "\\"
        filepath = configloader._get_config_path(node).replace("\\","/")

        ground_truth = os.path.join("tmp_dir",
                                    "configs",
                                    node.replace(".", "/") + ".yml").replace("\\","/")

        assert filepath == ground_truth

    def test_config_loader_load_correct_yaml(self, configloader):
        node = "input.test"
        data = {"input": "img",
                "output": "img"}
        create_config_yaml(node, data)

        config = configloader.get(node)

        for key in data.keys():
            assert data[key] == config[key]
