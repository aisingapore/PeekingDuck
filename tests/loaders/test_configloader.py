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
import shutil
import yaml

from peekingduck.loaders import ConfigLoader


CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
CONFIG_PATH = os.path.join(CURRENT_PATH, 'configs')


def create_configloader():
    config_loader = ConfigLoader(CURRENT_PATH)

    return config_loader


def create_config_yaml(node, data):
    node_type, node_name = node.split(".")

    node_config_path = os.path.join(CONFIG_PATH, node_type)
    os.makedirs(node_config_path)
    config_file = node_name + ".yml"
    full_path = os.path.join(node_config_path, config_file)

    with open(full_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    return CONFIG_PATH


class TestConfigLoader():

    def test_config_loader_returns_correct_config_filepath(self):

        node = 'type.node'
        config_loader = create_configloader()
        filepath = config_loader._get_config_path(node)

        ground_truth = os.path.join(CURRENT_PATH,
                                    "configs",
                                    node.replace(".", "/"))
        ground_truth = ground_truth + ".yml"

        assert filepath == ground_truth

    def test_config_loader_load_correct_yaml(self):
        node = "input.test"
        data = {"input": "img",
                "output": "img"}

        config_folder_dir = create_config_yaml(node, data)

        config_loader = create_configloader()
        config = config_loader.get(node)

        shutil.rmtree(config_folder_dir)

        for key in data.keys():
            assert data[key] == config[key]
