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

import os
import random
import string
import sys
import textwrap
from pathlib import Path
from unittest import TestCase

import pytest
import yaml
from click.testing import CliRunner

from peekingduck import __version__
from peekingduck.cli import cli

UNIQUE_SUFFIX = "".join(random.choice(string.ascii_lowercase) for _ in range(8))
CUSTOM_FOLDER_NAME = f"custom_nodes_{UNIQUE_SUFFIX}"

PKD_NODE_TYPE = "pkd_node_type"
PKD_NODE_NAME = "pkd_node_name"
PKD_NODE_NAME_2 = "pkd_node_name2"
PKD_NODE = f"{CUSTOM_FOLDER_NAME}.{PKD_NODE_TYPE}.{PKD_NODE_NAME}"
PKD_NODE_2 = f"{CUSTOM_FOLDER_NAME}.{PKD_NODE_TYPE}.{PKD_NODE_NAME_2}"
NODES = {"nodes": [PKD_NODE, PKD_NODE_2]}

PROJECT_DIR = Path("tmp_dir")
MODULE_DIR = PROJECT_DIR / "src"

DEFAULT_NODE_DIR = MODULE_DIR / "custom_nodes"
DEFAULT_NODE_CONFIG_DIR = DEFAULT_NODE_DIR / "configs"
CUSTOM_NODE_DIR = MODULE_DIR / CUSTOM_FOLDER_NAME
CUSTOM_NODE_CONFIG_DIR = CUSTOM_NODE_DIR / "configs"
CUSTOM_PKD_NODE_DIR = MODULE_DIR / CUSTOM_FOLDER_NAME / PKD_NODE_TYPE
CUSTOM_PKD_NODE_CONFIG_DIR = CUSTOM_NODE_CONFIG_DIR / PKD_NODE_TYPE
RUN_CONFIG_PATH = PROJECT_DIR / "run_config.yml"
YML = dict(nodes=["input.live", "model.yolo", "draw.bbox", "output.screen"])


def create_node_config(config_dir, node_name, config_text):
    with open(config_dir / f"{node_name}.yml", "w") as outfile:
        yaml.dump(config_text, outfile)


def create_node_python(node_dir, node_name, return_statement):
    with open(node_dir / f"{node_name}.py", "w") as outfile:
        content = textwrap.dedent(
            f"""\
            from peekingduck.pipeline.nodes.node import AbstractNode

            class Node(AbstractNode):
                def __init__(self, config, **kwargs):
                    super().__init__(config, node_path=__name__, **kwargs)

                def run(self, inputs):
                    return {{ {return_statement} }}
            """
        )
        outfile.write(content)


def create_run_config_yaml(nodes):
    # Inside PROJECT_DIR through usefixtures, use the leaf of the path
    with open(RUN_CONFIG_PATH.name, "w") as outfile:
        yaml.dump(nodes, outfile, default_flow_style=False)


def init_msg(node_name):
    return f"Initialising {node_name} node..."


def setup():
    sys.path.append(str(MODULE_DIR))

    relative_node_dir = CUSTOM_PKD_NODE_DIR.relative_to(CUSTOM_PKD_NODE_DIR.parts[0])
    relative_config_dir = CUSTOM_PKD_NODE_CONFIG_DIR.relative_to(
        CUSTOM_PKD_NODE_CONFIG_DIR.parts[0]
    )
    relative_node_dir.mkdir(parents=True)
    relative_config_dir.mkdir(parents=True)
    node_config_1 = {
        "input": ["none"],
        "output": ["test_output_1"],
        "resize": {"do_resizing": False},
    }
    node_config_2 = {"input": ["test_output_1"], "output": ["pipeline_end"]}

    create_run_config_yaml(NODES)
    create_node_python(relative_node_dir, PKD_NODE_NAME, "'test_output_1': None")
    create_node_python(relative_node_dir, PKD_NODE_NAME_2, "'pipeline_end': True")
    create_node_config(relative_config_dir, PKD_NODE_NAME, node_config_1)
    create_node_config(relative_config_dir, PKD_NODE_NAME_2, node_config_2)


@pytest.fixture
def tmp_project_dir():
    cwd = Path.cwd()
    (cwd / PROJECT_DIR).mkdir(parents=True)
    os.chdir(PROJECT_DIR)
    yield
    os.chdir(cwd)


@pytest.mark.usefixtures("tmp_dir", "tmp_project_dir")
class TestCli:
    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert f"version {__version__}" in result.output

    def test_init_default(self):
        parent_dir = Path.cwd().parent
        runner = CliRunner()
        with TestCase.assertLogs("peekingduck.cli.logger") as captured:
            result = runner.invoke(cli, ["init"])

            assert result.exit_code == 0
            assert "Creating custom nodes folder in" in captured.records[0].getMessage()
            assert "custom_nodes" in captured.records[0].getMessage()
            assert (parent_dir / DEFAULT_NODE_DIR).exists()
            assert (parent_dir / DEFAULT_NODE_CONFIG_DIR).exists()
            assert (parent_dir / RUN_CONFIG_PATH).exists()
            with open(parent_dir / RUN_CONFIG_PATH) as infile:
                TestCase().assertDictEqual(YML, yaml.safe_load(infile))

    def test_init_custom(self):
        parent_dir = Path.cwd().parent
        runner = CliRunner()
        with TestCase.assertLogs("peekingduck.cli.logger") as captured:
            result = runner.invoke(
                cli, ["init", "--custom_folder_name", CUSTOM_FOLDER_NAME]
            )

            assert result.exit_code == 0
            assert "Creating custom nodes folder in" in captured.records[0].getMessage()
            assert CUSTOM_FOLDER_NAME in captured.records[0].getMessage()
            assert (parent_dir / CUSTOM_NODE_DIR).exists()
            assert (parent_dir / CUSTOM_NODE_CONFIG_DIR).exists()
            assert (parent_dir / RUN_CONFIG_PATH).exists()
            with open(parent_dir / RUN_CONFIG_PATH) as infile:
                TestCase().assertDictEqual(YML, yaml.safe_load(infile))

    def test_run_default(self):
        setup()
        runner = CliRunner()
        with TestCase.assertLogs("peekingduck.cli.logger") as captured:
            result = runner.invoke(cli, ["run"])
            assert (
                "Successfully loaded run_config file."
                == captured.records[0].getMessage()
            )
            assert init_msg(PKD_NODE) == captured.records[1].getMessage()
            assert init_msg(PKD_NODE_2) == captured.records[2].getMessage()
            assert result.exit_code == 0

    def test_run_custom(self):
        setup()
        node_name = ".".join(PKD_NODE.split(".")[1:])
        config_update_value = "'do_resizing': True"
        config_update_cli = (
            f"{{'{node_name}': {{'resize': {{ {config_update_value} }} }} }}"
        )
        runner = CliRunner()

        with TestCase.assertLogs("peekingduck.cli.logger") as captured:
            result = runner.invoke(cli, ["run", "--node_config", config_update_cli])
            assert (
                "Successfully loaded run_config file."
                == captured.records[0].getMessage()
            )
            assert init_msg(PKD_NODE) == captured.records[1].getMessage()
            assert (
                f"Config for node {node_name} is updated to: {config_update_value}"
                == captured.records[2].getMessage()
            )
            assert init_msg(PKD_NODE_2) == captured.records[3].getMessage()
            assert result.exit_code == 0
