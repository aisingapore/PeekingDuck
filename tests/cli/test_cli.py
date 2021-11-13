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

import io
import math
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
RUN_CONFIG_PATH = Path("run_config.yml")
CUSTOM_RUN_CONFIG_PATH = Path("custom_dir") / "run_config.yml"
YML = dict(nodes=["input.live", "model.yolo", "draw.bbox", "output.screen"])

NODE_TYPES = ["input", "model", "dabble", "draw", "output"]
PKD_DIR = Path(__file__).resolve().parents[2] / "peekingduck"
PKD_CONFIG_DIR = PKD_DIR / "configs"
PKD_NODES_DIR = PKD_DIR / "pipeline" / "nodes"

with open(
    PKD_DIR.parent / "tests" / "data" / "user_inputs" / "create_node.yml"
) as infile:
    CREATE_NODE_INPUT = yaml.safe_load(infile.read())


@pytest.fixture(params=[0, 1])
def create_node_input_abort(request):
    yield (
        CREATE_NODE_INPUT["bad_paths"],
        CREATE_NODE_INPUT["bad_types"],
        CREATE_NODE_INPUT["bad_names"],
        CREATE_NODE_INPUT["good_paths"][request.param],
        CREATE_NODE_INPUT["good_types"][request.param],
        CREATE_NODE_INPUT["good_names"][request.param],
        CREATE_NODE_INPUT["proceed"]["reject"],
    )


@pytest.fixture(params=[0, 1])
def create_node_input_accept(request):
    yield (
        CREATE_NODE_INPUT["bad_paths"],
        CREATE_NODE_INPUT["bad_types"],
        CREATE_NODE_INPUT["bad_names"],
        CREATE_NODE_INPUT["good_paths"][request.param],
        CREATE_NODE_INPUT["good_types"][request.param],
        CREATE_NODE_INPUT["good_names"][request.param],
        CREATE_NODE_INPUT["proceed"]["accept"][request.param],
    )


def available_nodes_msg(type_name=None):
    output = io.StringIO()
    if type_name is None:
        node_types = NODE_TYPES
    else:
        node_types = [type_name]

    url_prefix = "https://peekingduck.readthedocs.io/en/stable/"
    url_postfix = ".html#"
    for node_type in node_types:
        node_names = [path.stem for path in (PKD_CONFIG_DIR / node_type).glob("*.yml")]
        max_length = len(max(node_names, key=len))
        print(f"\nPeekingDuck has the following {node_type} nodes:", file=output)
        for num, node_name in enumerate(node_names):
            idx = num + 1
            node_path = f"peekingduck.pipeline.nodes.{node_type}.{node_name}.Node"
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
            from peekingduck.pipeline.nodes.node import AbstractNode

            class Node(AbstractNode):
                def __init__(self, config, **kwargs):
                    super().__init__(config, node_path=__name__, **kwargs)

                def run(self, inputs):
                    return {{ {return_statement} }}
            """
        )
        outfile.write(content)


def create_run_config_yaml(nodes, custom_config_path):
    with open(
        CUSTOM_RUN_CONFIG_PATH if custom_config_path else RUN_CONFIG_PATH, "w"
    ) as outfile:
        yaml.dump(nodes, outfile, default_flow_style=False)


def get_custom_node_subpaths(node_subdir, node_type, node_name):
    return (
        f"{node_subdir}/configs/{node_type}/{node_name}.yml",
        f"{node_subdir}/{node_type}/{node_name}.py",
    )


def init_msg(node_name):
    return f"Initialising {node_name} node..."


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
        CUSTOM_RUN_CONFIG_PATH.parent.mkdir()
    node_config_1 = {
        "input": ["none"],
        "output": ["test_output_1"],
        "resize": {"do_resizing": False},
    }
    node_config_2 = {"input": ["test_output_1"], "output": ["pipeline_end"]}

    create_run_config_yaml(NODES, custom_config_path)
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


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def tmp_project_dir():
    cwd = Path.cwd()
    (cwd / PROJECT_DIR).mkdir(parents=True)
    os.chdir(PROJECT_DIR)
    yield
    os.chdir(cwd)


@pytest.mark.usefixtures("tmp_dir", "tmp_project_dir")
class TestCli:
    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        # not testing full message as .invoke() sets program name to cli
        # instead of peekingduck
        assert f"version {__version__}" in result.output

    def test_init_default(self, runner, parent_dir, cwd):
        with TestCase.assertLogs("peekingduck.cli.logger") as captured:
            result = runner.invoke(cli, ["init"])

            assert result.exit_code == 0
            assert (
                captured.records[0].getMessage()
                == f"Creating custom nodes folder in {cwd / 'src' / 'custom_nodes'}"
            )
            assert (parent_dir / DEFAULT_NODE_DIR).exists()
            assert (parent_dir / DEFAULT_NODE_CONFIG_DIR).exists()
            assert (cwd / RUN_CONFIG_PATH).exists()
            with open(cwd / RUN_CONFIG_PATH) as infile:
                TestCase().assertDictEqual(YML, yaml.safe_load(infile))

    def test_init_custom(self, runner, parent_dir, cwd):
        with TestCase.assertLogs("peekingduck.cli.logger") as captured:
            result = runner.invoke(
                cli, ["init", "--custom_folder_name", CUSTOM_FOLDER_NAME]
            )
            assert result.exit_code == 0
            assert (
                captured.records[0].getMessage()
                == f"Creating custom nodes folder in {cwd / 'src' / CUSTOM_FOLDER_NAME}"
            )
            assert (parent_dir / CUSTOM_NODE_DIR).exists()
            assert (parent_dir / CUSTOM_NODE_CONFIG_DIR).exists()
            assert (cwd / RUN_CONFIG_PATH).exists()
            with open(cwd / RUN_CONFIG_PATH) as infile:
                TestCase().assertDictEqual(YML, yaml.safe_load(infile))

    def test_run_default(self, runner):
        setup()
        with TestCase.assertLogs("peekingduck.cli.logger") as captured:
            result = runner.invoke(cli, ["run"])
            assert (
                captured.records[0].getMessage()
                == "Successfully loaded run_config file."
            )
            assert captured.records[1].getMessage() == init_msg(PKD_NODE)
            assert captured.records[2].getMessage() == init_msg(PKD_NODE_2)
            assert result.exit_code == 0

    def test_run_custom_path(self, runner):
        setup(True)
        with TestCase.assertLogs("peekingduck.cli.logger") as captured:
            result = runner.invoke(
                cli, ["run", "--config_path", CUSTOM_RUN_CONFIG_PATH]
            )
            assert (
                captured.records[0].getMessage()
                == "Successfully loaded run_config file."
            )
            assert captured.records[1].getMessage() == init_msg(PKD_NODE)
            assert captured.records[2].getMessage() == init_msg(PKD_NODE_2)
            assert result.exit_code == 0

    def test_run_custom_config(self, runner):
        setup()
        node_name = ".".join(PKD_NODE.split(".")[1:])
        config_update_value = "'do_resizing': True"
        config_update_cli = (
            f"{{'{node_name}': {{'resize': {{ {config_update_value} }} }} }}"
        )
        with TestCase.assertLogs("peekingduck.cli.logger") as captured:
            result = runner.invoke(cli, ["run", "--node_config", config_update_cli])
            assert (
                captured.records[0].getMessage()
                == "Successfully loaded run_config file."
            )
            assert captured.records[1].getMessage() == init_msg(PKD_NODE)
            assert (
                captured.records[2].getMessage()
                == f"Config for node {node_name} is updated to: {config_update_value}"
            )
            assert captured.records[3].getMessage() == init_msg(PKD_NODE_2)
            assert result.exit_code == 0

    def test_create_node_abort(self, runner, create_node_input_abort):
        (
            bad_paths,
            bad_types,
            bad_names,
            good_path,
            good_type,
            good_name,
            proceed,
        ) = create_node_input_abort
        result = runner.invoke(
            cli,
            ["create-node"],
            input=bad_paths
            + good_path
            + bad_types
            + good_type
            + bad_names
            + good_name
            + proceed,
        )
        # Count only substring we create so we are unaffected by click changes
        config_subpath, script_subpath = get_custom_node_subpaths(
            good_path.strip(), good_type.strip(), good_name.strip()
        )
        assert result.output.count("Path cannot") == bad_paths.count("\n")
        assert result.output.count("invalid choice") == bad_types.count("\n")
        assert result.output.count("Invalid node name") == bad_names.count("\n")
        assert result.output.count(config_subpath) == 1
        assert result.output.count(script_subpath) == 1
        assert result.output.count("Aborted!") == 1

    def test_create_node_accept(self, runner, create_node_input_accept):
        (
            bad_paths,
            bad_types,
            bad_names,
            good_path,
            good_type,
            good_name,
            proceed,
        ) = create_node_input_accept
        result = runner.invoke(
            cli,
            ["create-node"],
            input=bad_paths
            + good_path
            + bad_types
            + good_type
            + bad_names
            + good_name
            + proceed,
        )
        # Count only substring we create so we are unaffected by click changes
        config_subpath, script_subpath = get_custom_node_subpaths(
            good_path.strip(), good_type.strip(), good_name.strip()
        )
        assert result.output.count("Path cannot") == bad_paths.count("\n")
        assert result.output.count("invalid choice") == bad_types.count("\n")
        assert result.output.count("Invalid node name") == bad_names.count("\n")
        assert result.output.count(config_subpath) == 1
        assert result.output.count(script_subpath) == 1
        assert result.output.count("Created node!") == 1

        node_subdir = good_path.strip()
        node_type = good_type.strip()
        node_name = good_name.strip()
        cwd = Path.cwd()
        config_path = cwd / node_subdir / "configs" / node_type / f"{node_name}.yml"
        script_path = cwd / node_subdir / node_type / f"{node_name}.py"
        assert config_path.exists()
        assert script_path.exists()

        with open(config_path) as actual_file, open(
            PKD_CONFIG_DIR / "node_template.yml"
        ) as expected_file:
            assert actual_file.read() == expected_file.read()

        with open(script_path) as actual_file, open(
            PKD_NODES_DIR / "node_template.py"
        ) as expected_file:
            assert actual_file.read() == expected_file.read()

    def test_create_node_duplicate_node_name(self, runner, create_node_input_accept):
        _, _, _, good_path, good_type, good_name, proceed = create_node_input_accept

        node_subdir = good_path.strip()
        node_type = good_type.strip()
        node_name = good_name.strip()
        node_name_2 = "available_node_name"
        name_input = f"{good_name}{node_name_2}\n"
        setup_custom_node(node_subdir, node_type, node_name)
        result = runner.invoke(
            cli,
            ["create-node"],
            input=good_path + good_type + name_input + proceed,
        )
        # Only check the "Node name exists" message, others are checked by
        # previous tests.
        assert result.output.count("Node name already exists!") == 1

        cwd = Path.cwd()
        config_path = cwd / node_subdir / "configs" / node_type / f"{node_name_2}.yml"
        script_path = cwd / node_subdir / node_type / f"{node_name_2}.py"
        assert config_path.exists()
        assert script_path.exists()

        with open(config_path) as actual_file, open(
            PKD_CONFIG_DIR / "node_template.yml"
        ) as expected_file:
            assert actual_file.read() == expected_file.read()

        with open(script_path) as actual_file, open(
            PKD_NODES_DIR / "node_template.py"
        ) as expected_file:
            assert actual_file.read() == expected_file.read()

    def test_create_node_cli_options_abort(self, runner, create_node_input_abort):
        (
            bad_paths,
            bad_types,
            bad_names,
            good_path,
            good_type,
            good_name,
            proceed,
        ) = create_node_input_abort
        result = runner.invoke(
            cli,
            [
                "create-node",
                "--node_subdir",
                "../some/path",
                "--node_type",
                "some type",
                "--node_name",
                "some name",
            ],
            input=bad_paths
            + good_path
            + bad_types
            + good_type
            + bad_names
            + good_name
            + proceed,
        )
        # Count only substring we create so we are unaffected by click changes
        config_subpath, script_subpath = get_custom_node_subpaths(
            good_path.strip(), good_type.strip(), good_name.strip()
        )
        assert result.output.count("Path cannot") == bad_paths.count("\n") + 1
        assert result.output.count("invalid choice") == bad_types.count("\n") + 1
        assert result.output.count("Invalid node name") == bad_names.count("\n") + 1
        assert result.output.count(config_subpath) == 1
        assert result.output.count(script_subpath) == 1
        assert result.output.count("Aborted!") == 1

    def test_create_node_cli_options_accept(self, runner, create_node_input_accept):
        _, _, _, good_path, good_type, good_name, proceed = create_node_input_accept
        node_subdir = good_path.strip()
        node_type = good_type.strip()
        node_name = good_name.strip()
        # Creates only using CLI options with minimal user input
        result = runner.invoke(
            cli,
            [
                "create-node",
                "--node_subdir",
                node_subdir,
                "--node_type",
                node_type,
                "--node_name",
                node_name,
            ],
            input=proceed,
        )
        config_subpath, script_subpath = get_custom_node_subpaths(
            node_subdir, node_type, node_name
        )
        assert result.output.count(config_subpath) == 1
        assert result.output.count(script_subpath) == 1
        assert result.output.count("Created node!") == 1

        cwd = Path.cwd()
        config_path = cwd / node_subdir / "configs" / node_type / f"{node_name}.yml"
        script_path = cwd / node_subdir / node_type / f"{node_name}.py"
        assert config_path.exists()
        assert script_path.exists()

        with open(config_path) as actual_file, open(
            PKD_CONFIG_DIR / "node_template.yml"
        ) as expected_file:
            assert actual_file.read() == expected_file.read()

        with open(script_path) as actual_file, open(
            PKD_NODES_DIR / "node_template.py"
        ) as expected_file:
            assert actual_file.read() == expected_file.read()

    def test_nodes_all(self, runner):
        result = runner.invoke(cli, ["nodes"])
        print(result.exception)
        print(result.output)
        assert result.exit_code == 0
        assert result.output == available_nodes_msg()

    def test_nodes_single(self, runner):
        for node_type in NODE_TYPES:
            result = runner.invoke(cli, ["nodes", node_type])
            assert result.exit_code == 0
            assert result.output == available_nodes_msg(node_type)
