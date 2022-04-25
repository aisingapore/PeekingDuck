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
import random
import string
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest import TestCase

import pytest
import yaml
from click.testing import CliRunner

from peekingduck import __version__
from peekingduck.cli import cli
from peekingduck.declarative_loader import PEEKINGDUCK_NODE_TYPES

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
PIPELINE_PATH = Path("pipeline_config.yml")
CUSTOM_PIPELINE_PATH = Path("custom_dir") / "pipeline_config.yml"
YML = dict(
    nodes=[
        {
            "input.visual": {
                "source": "https://storage.googleapis.com/peekingduck/videos/wave.mp4"
            }
        },
        "model.posenet",
        "draw.poses",
        "output.screen",
    ]
)

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


def create_node_config(config_dir, node_name, config_text):
    with open(config_dir / f"{node_name}.yml", "w") as outfile:
        yaml.dump(config_text, outfile)


def create_node_python(node_dir, node_name, return_statement):
    with open(node_dir / f"{node_name}.py", "w") as outfile:
        content = textwrap.dedent(
            f"""\
            from peekingduck.pipeline.nodes.abstract_node import AbstractNode

            class Node(AbstractNode):
                def __init__(self, config, **kwargs):
                    super().__init__(config, node_path=__name__, **kwargs)

                def run(self, inputs):
                    return {{ {return_statement} }}
            """
        )
        outfile.write(content)


def create_pipeline_yaml(nodes, custom_config_path):
    with open(
        CUSTOM_PIPELINE_PATH if custom_config_path else PIPELINE_PATH, "w"
    ) as outfile:
        yaml.dump(nodes, outfile, default_flow_style=False)


def get_custom_node_subpaths(node_subdir, node_type, node_name):
    return (
        str(Path(node_subdir) / "configs" / node_type / f"{node_name}.yml"),
        str(Path(node_subdir) / node_type / f"{node_name}.py"),
    )


def init_msg(node_name):
    return f"Initializing {node_name} node..."


def setup_custom_node(node_subdir, node_type, node_name):
    cwd = Path.cwd()
    config_dir = cwd / node_subdir / "configs" / node_type
    script_dir = cwd / node_subdir / node_type
    config_dir.mkdir(parents=True, exist_ok=True)
    script_dir.mkdir(parents=True, exist_ok=True)

    (config_dir / f"{node_name}.yml").touch()
    (script_dir / f"{node_name}.py").touch()


def setup(custom_config_path=False):
    sys.path.append(str(MODULE_DIR))

    relative_node_dir = CUSTOM_PKD_NODE_DIR.relative_to(CUSTOM_PKD_NODE_DIR.parts[0])
    relative_config_dir = CUSTOM_PKD_NODE_CONFIG_DIR.relative_to(
        CUSTOM_PKD_NODE_CONFIG_DIR.parts[0]
    )
    relative_node_dir.mkdir(parents=True)
    relative_config_dir.mkdir(parents=True)
    if custom_config_path:
        CUSTOM_PIPELINE_PATH.parent.mkdir()
    node_config_1 = {
        "input": ["none"],
        "output": ["test_output_1"],
        "resize": {"do_resizing": False},
    }
    node_config_2 = {"input": ["test_output_1"], "output": ["pipeline_end"]}

    create_pipeline_yaml(NODES, custom_config_path)
    create_node_python(relative_node_dir, PKD_NODE_NAME, "'test_output_1': None")
    create_node_python(relative_node_dir, PKD_NODE_NAME_2, "'pipeline_end': True")
    create_node_config(relative_config_dir, PKD_NODE_NAME, node_config_1)
    create_node_config(relative_config_dir, PKD_NODE_NAME_2, node_config_2)


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

    def test_init_default(self, parent_dir, cwd):
        with TestCase.assertLogs("peekingduck.cli.logger") as captured:
            result = CliRunner().invoke(cli, ["init"])

            assert result.exit_code == 0
            assert (
                captured.records[0].getMessage()
                == f"Creating custom nodes folder in {cwd / 'src' / 'custom_nodes'}"
            )
            assert (parent_dir / DEFAULT_NODE_DIR).exists()
            assert (parent_dir / DEFAULT_NODE_CONFIG_DIR).exists()
            assert (cwd / PIPELINE_PATH).exists()
            with open(cwd / PIPELINE_PATH) as infile:
                config_file = yaml.safe_load(infile)
                TestCase().assertDictEqual(YML, config_file)

    def test_init_custom(self, parent_dir, cwd):
        with TestCase.assertLogs("peekingduck.cli.logger") as captured:
            result = CliRunner().invoke(
                cli, ["init", "--custom_folder_name", CUSTOM_FOLDER_NAME]
            )
            assert result.exit_code == 0
            assert (
                captured.records[0].getMessage()
                == f"Creating custom nodes folder in {cwd / 'src' / CUSTOM_FOLDER_NAME}"
            )
            assert (parent_dir / CUSTOM_NODE_DIR).exists()
            assert (parent_dir / CUSTOM_NODE_CONFIG_DIR).exists()
            assert (cwd / PIPELINE_PATH).exists()
            with open(cwd / PIPELINE_PATH) as infile:
                TestCase().assertDictEqual(YML, yaml.safe_load(infile))

    def test_run_default(self):
        setup()
        with TestCase.assertLogs("peekingduck.cli.logger") as captured:
            result = CliRunner().invoke(cli, ["run"])
            assert (
                captured.records[0].getMessage() == "Successfully loaded pipeline file."
            )
            assert captured.records[1].getMessage() == init_msg(PKD_NODE)
            assert captured.records[2].getMessage() == init_msg(PKD_NODE_2)
            assert result.exit_code == 0

    def test_run_custom_path(self):
        setup(True)
        with TestCase.assertLogs("peekingduck.cli.logger") as captured:
            result = CliRunner().invoke(
                cli, ["run", "--config_path", CUSTOM_PIPELINE_PATH]
            )
            assert (
                captured.records[0].getMessage() == "Successfully loaded pipeline file."
            )
            assert captured.records[1].getMessage() == init_msg(PKD_NODE)
            assert captured.records[2].getMessage() == init_msg(PKD_NODE_2)
            assert result.exit_code == 0

    def test_run_custom_config(self):
        setup()
        node_name = ".".join(PKD_NODE.split(".")[1:])
        config_update_value = "'do_resizing': True"
        config_update_cli = (
            f"{{'{node_name}': {{'resize': {{ {config_update_value} }} }} }}"
        )
        with TestCase.assertLogs("peekingduck.cli.logger") as captured:
            result = CliRunner().invoke(
                cli, ["run", "--node_config", config_update_cli]
            )
            assert (
                captured.records[0].getMessage() == "Successfully loaded pipeline file."
            )
            assert captured.records[1].getMessage() == init_msg(PKD_NODE)
            assert (
                captured.records[2].getMessage()
                == f"Config for node {node_name} is updated to: {config_update_value}"
            )
            assert captured.records[3].getMessage() == init_msg(PKD_NODE_2)
            assert result.exit_code == 0

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
        test_config_path = tmp_dir / "test_config.yml"
        nodes = {
            "nodes": [{"input.visual": {"source": "PeekingDuck/tests/data/images"}}]
        }
        with open(test_config_path, "w") as outfile:
            yaml.dump(nodes, outfile, default_flow_style=False)

        unit_test_run_dir = Path(__file__).parents[3]
        print(f"unit_test_run_dir={unit_test_run_dir}")

        # run unit test
        os.chdir(unit_test_run_dir)
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

    def test_num_iter(self):
        setup()
        with TestCase.assertLogs("peekingduck.cli.logger") as captured:
            n = 50  # run test for 50 iterations
            result = CliRunner().invoke(cli, ["run", "--num_iter", n])
            assert (
                captured.records[0].getMessage() == "Successfully loaded pipeline file."
            )
            assert captured.records[1].getMessage() == init_msg(PKD_NODE)
            assert captured.records[2].getMessage() == init_msg(PKD_NODE_2)
            assert (
                captured.records[3].getMessage() == f"Run pipeline for {n} iterations"
            )
            assert result.exit_code == 0
