# Copyright 2022 AI Singapore
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

from pathlib import Path

import pytest
import yaml

from peekingduck.config_loader import ConfigLoader


@pytest.fixture
def config_loader():
    config_loader = ConfigLoader(Path.cwd() / "tmp_dir")

    return config_loader


def create_config_yaml(node, data):
    node_type, node_name = node.split(".")
    config_dir = Path.cwd() / "tmp_dir" / "configs"

    node_config_dir = config_dir / node_type
    node_config_dir.mkdir(parents=True)

    full_path = node_config_dir / f"{node_name}.yml"

    with open(full_path, "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


@pytest.mark.usefixtures("tmp_dir")
class TestConfigLoader:
    def test_config_loader_returns_correct_config_file_path(self, config_loader):
        node = "type.node"
        # .replace("\\","/") for windows where os.path.join uses "\\"
        file_path = str(config_loader._get_config_path(node)).replace("\\", "/")

        ground_truth = str(
            Path.cwd() / "tmp_dir" / "configs" / f"{node.replace('.', '/')}.yml"
        ).replace("\\", "/")

        assert file_path == ground_truth

    def test_config_loader_load_correct_yaml(self, config_loader):
        node = "input.test"
        data = {"input": "img", "output": "img"}
        create_config_yaml(node, data)

        config = config_loader.get(node)

        for key in data.keys():
            assert data[key] == config[key]
