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

import sys
from logging import LogRecord
from pathlib import Path
from unittest import TestCase

import pytest
import yaml
from click.testing import CliRunner

from peekingduck.cli import cli

DEFAULT_NODES = ["input.visual", "model.yolo", "draw.bbox", "output.screen"]
GOOD_SUBDIR = "custom_nodes"
GOOD_TYPE = "dabble"
GOOD_NAME = "name"
PROJECT_DIR = Path("tmp_dir")

PKD_DIR = Path(__file__).resolve().parents[2] / "peekingduck"
PKD_CONFIG_DIR = PKD_DIR / "configs"
PKD_NODES_DIR = PKD_DIR / "pipeline" / "nodes"

with open(
    PKD_DIR.parent / "tests" / "data" / "user_inputs" / "create_node.yml"
) as infile:
    CREATE_NODE_INPUT = yaml.safe_load(infile.read())

with open(
    PKD_DIR.parent / "tests" / "data" / "user_configs" / "create_node.yml"
) as infile:
    CREATE_NODE_CONFIG = yaml.safe_load(infile.read())


@pytest.fixture(params=[0, 1])
def create_node_input_abort(request):
    # Windows has a different absolute path format
    bad_paths = "bad_paths_win" if sys.platform == "win32" else "bad_paths"
    yield (
        CREATE_NODE_INPUT[bad_paths],
        CREATE_NODE_INPUT["bad_types"],
        CREATE_NODE_INPUT["bad_names"],
        CREATE_NODE_INPUT["good_paths"][request.param],
        CREATE_NODE_INPUT["good_types"][request.param],
        CREATE_NODE_INPUT["good_names"][request.param],
        CREATE_NODE_INPUT["proceed"]["reject"],
    )


@pytest.fixture(params=[0, 1])
def create_node_input_accept(request):
    bad_paths = "bad_paths_win" if sys.platform == "win32" else "bad_paths"
    yield (
        CREATE_NODE_INPUT[bad_paths],
        CREATE_NODE_INPUT["bad_types"],
        CREATE_NODE_INPUT["bad_names"],
        CREATE_NODE_INPUT["good_paths"][request.param],
        CREATE_NODE_INPUT["good_types"][request.param],
        CREATE_NODE_INPUT["good_names"][request.param],
        CREATE_NODE_INPUT["proceed"]["accept"][request.param],
    )


@pytest.fixture
def cwd():
    return Path.cwd()


def get_custom_node_subpaths(node_subdir, node_type, node_name):
    return (
        str(Path(node_subdir) / "configs" / node_type / f"{node_name}.yml"),
        str(Path(node_subdir) / node_type / f"{node_name}.py"),
    )


def setup_custom_node(node_subdir, node_type, node_name):
    cwd = Path.cwd()
    config_dir = cwd / node_subdir / "configs" / node_type
    script_dir = cwd / node_subdir / node_type
    config_dir.mkdir(parents=True, exist_ok=True)
    script_dir.mkdir(parents=True, exist_ok=True)

    (config_dir / f"{node_name}.yml").touch()
    (script_dir / f"{node_name}.py").touch()


@pytest.mark.usefixtures("tmp_dir", "tmp_project_dir")
class TestCliCreateNode:
    def test_abort(self, create_node_input_abort):
        (
            bad_paths,
            bad_types,
            bad_names,
            good_path,
            good_type,
            good_name,
            proceed,
        ) = create_node_input_abort
        result = CliRunner().invoke(
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
        assert result.output.count("is not one of") == bad_types.count("\n")
        assert result.output.count("Invalid node name") == bad_names.count("\n")
        assert result.output.count(config_subpath) == 1
        assert result.output.count(script_subpath) == 1
        assert result.output.count("Aborted!") == 1

    def test_accept(self, create_node_input_accept):
        (
            bad_paths,
            bad_types,
            bad_names,
            good_path,
            good_type,
            good_name,
            proceed,
        ) = create_node_input_accept
        result = CliRunner().invoke(
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
        assert result.output.count("is not one of") == bad_types.count("\n")
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
            lines = expected_file.readlines()
            # Ensuring start exists/is valid is not done here since we expect
            # it to always be valid
            for i, line in enumerate(lines):
                if line.startswith('"'):
                    start = i
                    break
            assert actual_file.readlines() == lines[start:]

    def test_duplicate_node_name(self, create_node_input_accept):
        _, _, _, good_path, good_type, good_name, proceed = create_node_input_accept

        node_subdir = good_path.strip()
        node_type = good_type.strip()
        node_name = good_name.strip()
        node_name_2 = "available_node_name"
        name_input = f"{good_name}{node_name_2}\n"
        setup_custom_node(node_subdir, node_type, node_name)
        result = CliRunner().invoke(
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
            lines = expected_file.readlines()
            for i, line in enumerate(lines):
                if line.startswith('"'):
                    start = i
                    break
            assert actual_file.readlines() == lines[start:]

    def test_cli_options_abort(self, create_node_input_abort):
        (
            bad_paths,
            bad_types,
            bad_names,
            good_path,
            good_type,
            good_name,
            proceed,
        ) = create_node_input_abort
        result = CliRunner().invoke(
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
        assert result.output.count("is not one of") == bad_types.count("\n") + 1
        assert result.output.count("Invalid node name") == bad_names.count("\n") + 1
        assert result.output.count(config_subpath) == 1
        assert result.output.count(script_subpath) == 1
        assert result.output.count("Aborted!") == 1

    def test_cli_options_accept(self, create_node_input_accept):
        _, _, _, good_path, good_type, good_name, proceed = create_node_input_accept
        node_subdir = good_path.strip()
        node_type = good_type.strip()
        node_name = good_name.strip()
        # Creates only using CLI options with minimal user input
        result = CliRunner().invoke(
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
            lines = expected_file.readlines()
            for i, line in enumerate(lines):
                if line.startswith('"'):
                    start = i
                    break
            assert actual_file.readlines() == lines[start:]

    def test_poorly_formatted_config_file(self, cwd):
        no_top_level_key = cwd / "pipeline_no_top_level_key.yml"
        wrong_top_level_key = cwd / "pipeline_wrong_top_level_key.yml"
        with open(no_top_level_key, "w") as outfile:
            yaml.dump(DEFAULT_NODES, outfile)
        with open(wrong_top_level_key, "w") as outfile:
            yaml.dump({"asdf": DEFAULT_NODES}, outfile)

        for path in (no_top_level_key, wrong_top_level_key):
            # This error originates from DeclarativeLoader
            with pytest.raises(ValueError) as excinfo:
                CliRunner().invoke(
                    cli,
                    ["create-node", "--config_path", path.name],
                    catch_exceptions=False,
                )
            assert "has an invalid structure. Missing top-level 'nodes' key." in str(
                excinfo.value
            )

    def test_no_nodes_pipeline_file(self, cwd):
        no_nodes = cwd / "pipeline_no_nodes.yml"
        with open(no_nodes, "w") as outfile:
            data = {"nodes": None}
            # ``yaml`` will create 'nodes: null' by default. Manually replace
            # it to just 'nodes: '. Surprisingly ``yaml`` will load this
            # without error
            outfile.write(yaml.safe_dump(data).replace("null", ""))

        # This error originates from DeclarativeLoader
        with pytest.raises(ValueError) as excinfo:
            CliRunner().invoke(
                cli,
                ["create-node", "--config_path", no_nodes.name],
                catch_exceptions=False,
            )
        assert "does not contain any nodes!" in str(excinfo.value)

    @pytest.mark.parametrize(
        "extra_options",
        [
            ["node_subdir"],
            ["node_type"],
            ["node_name"],
            ["node_subdir", "node_type", "node_name"],
        ],
    )
    def test_invalid_cli_options(self, extra_options):
        """Tests cases when at least one `node_` related option is used with
        `config_path` and when all three `node_` related options are used with
        `config_path`.
        """
        extra_args = [[f"--{option}", "value"] for option in extra_options]
        with pytest.raises(ValueError) as excinfo:
            CliRunner().invoke(
                cli,
                ["create-node", "--config_path", "value"]
                + [arg for arg_pair in extra_args for arg in arg_pair],
                catch_exceptions=False,
            )
        assert (
            "--config_path cannot be use with --node_subdir, --node_type, or "
            "--node_name!"
        ) in str(excinfo.value)

    def test_missing_config_file(self, cwd):
        config_file = "missing_file.yml"
        with pytest.raises(FileNotFoundError) as excinfo:
            CliRunner().invoke(
                cli,
                ["create-node", "--config_path", config_file],
                catch_exceptions=False,
            )
        assert f"Config file '{config_file}' is not found at" in str(excinfo.value)

    def test_no_custom_nodes(self, cwd):
        """Tests when the pipeline file doesn't contain any custom nodes, so
        there's nothing to create.
        """
        pipeline_file = "pipeline_default_nodes_only.yml"
        default_nodes_only = cwd / pipeline_file
        with open(default_nodes_only, "w") as outfile:
            yaml.dump({"nodes": DEFAULT_NODES}, outfile)
        with pytest.raises(ValueError) as excinfo:
            CliRunner().invoke(
                cli,
                ["create-node", "--config_path", pipeline_file],
                catch_exceptions=False,
            )
            assert (
                f"Config file '{pipeline_file}' does not contain custom nodes!"
                == str(excinfo.value)
            )

    def test_invalid_custom_node_string(self, cwd):
        """The custom nodes declared in the pipeline file only contains poor
        formatting. So all will be skipped.
        """
        bad_paths = [
            f"{node_subdir}.{GOOD_TYPE}.{GOOD_NAME}"
            for node_subdir in CREATE_NODE_CONFIG[
                "bad_config_paths_win"
                if sys.platform == "win32"
                else "bad_config_paths"
            ]
        ]
        bad_types = [
            f"{GOOD_SUBDIR}.{node_type}.{GOOD_NAME}"
            for node_type in CREATE_NODE_CONFIG["bad_config_types"]
        ]
        bad_names = [
            f"{GOOD_SUBDIR}.{GOOD_TYPE}.{node_name}"
            for node_name in CREATE_NODE_CONFIG["bad_config_names"]
        ]
        pipeline_file = "pipeline_invalid_custom_node_string.yml"
        default_nodes_only = cwd / pipeline_file
        with open(default_nodes_only, "w") as outfile:
            # Create a "challenging" file, with some config overrides
            data = {
                "nodes": [
                    {"input.visual": {"source": 0}},
                    {"model.yolo": {"model_type": "v4"}},
                    "draw.bbox",
                ]
                + bad_paths
                + bad_types
                + bad_names
                + ["output.screen"]
            }
            yaml.dump(data, outfile)
        with TestCase.assertLogs("peekingduck.cli.logger") as captured:
            CliRunner().invoke(cli, ["create-node", "--config_path", pipeline_file])
            offset = 2  # First 2 message is about info about loading config
            counter = 0
            for node_string in bad_paths:
                assert (
                    f"{node_string} contains invalid formatting: 'Path cannot be "
                    "absolute!'. Skipping..."
                ) == captured.records[offset + counter].getMessage()
                counter += 1
            for node_string in bad_types:
                # Invalid type error message begins with a 'user_input' so we
                # just check for the presence of the double single quote
                assert (
                    f"{node_string} contains invalid formatting: ''"
                ) in captured.records[offset + counter].getMessage()
                counter += 1
            for node_string in bad_names:
                assert (
                    f"{node_string} contains invalid formatting: 'Invalid node "
                    "name!'. Skipping..."
                ) == captured.records[offset + counter].getMessage()
                counter += 1

    def test_create_nodes_from_config_success(self, cwd):
        """The custom nodes declared in the pipeline file only contains poor
        formatting. So all will be skipped.
        """
        node_string = f"{GOOD_SUBDIR}.{GOOD_TYPE}.{GOOD_NAME}"
        created_config_path = (
            cwd / "src" / GOOD_SUBDIR / "configs" / GOOD_TYPE / f"{GOOD_NAME}.yml"
        )
        created_script_path = cwd / "src" / GOOD_SUBDIR / GOOD_TYPE / f"{GOOD_NAME}.py"
        pipeline_file = "pipeline_invalid_custom_node_string.yml"
        default_nodes_only = cwd / pipeline_file
        with open(default_nodes_only, "w") as outfile:
            # Create a "challenging" file, with some config overrides
            data = {
                "nodes": [
                    {"input.visual": {"source": 0}},
                    {"model.yolo": {"model_type": "v4"}},
                    "draw.bbox",
                    node_string,
                    "output.screen",
                ]
            }
            yaml.dump(data, outfile)
        with TestCase.assertLogs("peekingduck.cli.logger") as captured:
            CliRunner().invoke(cli, ["create-node", "--config_path", pipeline_file])
            # First 2 message is about info about loading config
            assert (
                f"Creating files for {node_string}:\n\t"
                f"Config file: {created_config_path}\n\t"
                f"Script file: {created_script_path}"
            ) == captured.records[2].getMessage()

    def test_create_nodes_from_config_duplicate_node_name(self, cwd):
        """The custom nodes declared in the pipeline file only contains poor
        formatting. So all will be skipped.
        """
        node_string = f"{GOOD_SUBDIR}.{GOOD_TYPE}.{GOOD_NAME}"
        pipeline_file = "pipeline_invalid_custom_node_string.yml"
        default_nodes_only = cwd / pipeline_file
        with open(default_nodes_only, "w") as outfile:
            # Create a "challenging" file, with some config overrides
            data = {
                "nodes": [
                    {"input.visual": {"source": 0}},
                    {"model.yolo": {"model_type": "v4"}},
                    "draw.bbox",
                    node_string,
                    "output.screen",
                ]
            }
            yaml.dump(data, outfile)
        # Create the node first so we trigger the duplicate name warning
        CliRunner().invoke(cli, ["create-node", "--config_path", pipeline_file])
        with TestCase.assertLogs("peekingduck.cli.logger") as captured:
            CliRunner().invoke(cli, ["create-node", "--config_path", pipeline_file])
            # First 2 message is about info about loading config
            assert (
                f"{node_string} contains invalid formatting: 'Node name already "
                "exists!'. Skipping..."
            ) == captured.records[2].getMessage()
