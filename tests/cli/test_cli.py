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

import io
import math
import os
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from peekingduck import __version__
from peekingduck.cli import cli
from peekingduck.declarative_loader import PEEKINGDUCK_NODE_TYPES

PKD_DIR = Path(__file__).resolve().parents[2] / "peekingduck"
PKD_CONFIG_DIR = PKD_DIR / "configs"


def available_nodes_msg(type_name=None):
    def len_enumerate(item):
        return int(math.log10(item[0] + 1)) + len(item[1])

    output = io.StringIO()
    if type_name is None:
        node_types = PEEKINGDUCK_NODE_TYPES
    else:
        node_types = [type_name]

    url_prefix = "https://peekingduck.readthedocs.io/en/stable/nodes/"
    url_postfix = ".html#module-"
    for node_type in node_types:
        node_names = [path.stem for path in (PKD_CONFIG_DIR / node_type).glob("*.yml")]
        max_length = len_enumerate(max(enumerate(node_names), key=len_enumerate))
        print(f"\nPeekingDuck has the following {node_type} nodes:", file=output)
        for num, node_name in enumerate(node_names):
            idx = num + 1
            node_path = f"{node_type}.{node_name}"
            url = f"{url_prefix}{node_path}{url_postfix}{node_path}"
            node_width = max_length + 1 - int(math.log10(idx))
            print(f"{idx}:{node_name: <{node_width}}Info: {url}", file=output)
    print("\n", file=output)
    content = output.getvalue()
    output.close()
    return content


@pytest.fixture
def cwd():
    return Path.cwd()


@pytest.fixture
def parent_dir():
    return Path.cwd().parent


@pytest.mark.usefixtures("tmp_dir", "tmp_project_dir")
class TestCli:
    def test_version(self):
        result = CliRunner().invoke(cli, ["--version"])

        assert result.exit_code == 0
        # not testing full message as .invoke() sets program name to cli
        # instead of peekingduck
        assert f"version {__version__}" in result.output

    def test_nodes_all(self):
        result = CliRunner().invoke(cli, ["nodes"])
        print(result.exception)
        print(result.output)
        assert result.exit_code == 0
        assert result.output == available_nodes_msg()

    def test_nodes_single(self):
        for node_type in PEEKINGDUCK_NODE_TYPES:
            result = CliRunner().invoke(cli, ["nodes", node_type])
            assert result.exit_code == 0
            assert result.output == available_nodes_msg(node_type)

    def test_main_py_log_level_debug(self):
        # setup unit test env
        tmp_dir = Path.cwd()
        print(f"\ntmp_dir={tmp_dir}")
        unit_test_run_dir = Path(__file__).parents[3]
        print(f"unit_test_run_dir={unit_test_run_dir}")
        nodes = {
            "nodes": [
                {
                    "input.visual": {
                        "source": f"{unit_test_run_dir}/PeekingDuck/tests/data/images"
                    }
                }
            ]
        }
        os.chdir(unit_test_run_dir)
        # test_config_path = tmp_dir / "test_config.yml"
        test_config_path = "test_config.yml"
        with open(test_config_path, "w") as outfile:
            yaml.dump(nodes, outfile, default_flow_style=False)

        # run unit test
        cmd = [
            "python",
            "PeekingDuck",
            "--log_level",
            "debug",
            "--config_path",
            f"{test_config_path}",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        out, _ = proc.communicate()
        out_str = out.decode("utf-8")
        print(out_str)
        exit_status = proc.returncode
        assert "DEBUG" in out_str
        assert exit_status == 0
