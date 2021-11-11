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

import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
import yaml

from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.runner import Runner

PKD_NODE_TYPE = "pkd_node_type"
PKD_NODE_NAME = "pkd_node_name"
PKD_NODE_NAME_2 = "pkd_node_name2"
PKD_NODE = f"{PKD_NODE_TYPE}.{PKD_NODE_NAME}"
PKD_NODE_2 = f"{PKD_NODE_TYPE}.{PKD_NODE_NAME_2}"
NODES = {"nodes": [PKD_NODE, PKD_NODE_2]}

MODULE_DIR = Path("tmp_dir")
RUN_CONFIG_PATH = MODULE_DIR / "run_config.yml"
CUSTOM_DIR = MODULE_DIR / "custom_nodes"
CUSTOM_CONFIG_DIR = MODULE_DIR / "configs" / PKD_NODE_TYPE
PKD_NODE_DIR = MODULE_DIR / PKD_NODE_TYPE
CONFIG_UPDATES_CLI = "{'input.live': {'resize': {'do_resizing': True}}}"


class MockedNode(AbstractNode):
    def __init__(self, config):
        super().__init__(config, node_path=PKD_NODE, pkdbasedir=str(MODULE_DIR))

    def run(self, inputs):
        output = {
            self.outputs[idx]: f"test_output_{idx}"
            for idx, _ in enumerate(self.outputs)
        }
        return output


def create_node_config(config_dir, node_name):
    config_text = {"root": None, "input": ["none"], "output": ["pipeline_end"]}
    with open(config_dir / f"{node_name}.yml", "w") as fp:
        yaml.dump(config_text, fp)


def create_run_config_yaml(nodes):
    with open(RUN_CONFIG_PATH, "w") as outfile:
        yaml.dump(nodes, outfile, default_flow_style=False)


def setup():
    sys.path.append(str(Path.cwd() / MODULE_DIR))
    PKD_NODE_DIR.mkdir(parents=True)
    create_run_config_yaml(NODES)


def get_pipeline_with_default_node_names():
    mock_node = mock.Mock()
    mock_node.inputs = ["none"]
    mock_node.name = f"peekingduck.pipeline.nodes.{PKD_NODE}"
    mock_node.node_name = PKD_NODE

    mock_pipeline = mock.Mock()
    mock_pipeline.nodes = [mock_node]

    return mock_pipeline


def replace_declarativeloader_get_pipeline():
    mock_pipeline = mock.Mock()
    mock_pipeline.nodes = []

    return mock_pipeline


def replace_pipeline_check_pipe(node):
    pass


def replace_pipeline(node):
    pass


@pytest.fixture
def runner():
    setup()
    with mock.patch(
        "peekingduck.declarative_loader.DeclarativeLoader.get_pipeline",
        wraps=replace_declarativeloader_get_pipeline,
    ):
        test_runner = Runner(RUN_CONFIG_PATH, CONFIG_UPDATES_CLI, CUSTOM_DIR)

        return test_runner


@pytest.fixture
def runner_with_default_node_names():
    setup()
    with mock.patch(
        "peekingduck.declarative_loader.DeclarativeLoader.get_pipeline",
        wraps=get_pipeline_with_default_node_names,
    ):
        test_runner = Runner(RUN_CONFIG_PATH, CONFIG_UPDATES_CLI, CUSTOM_DIR)

        return test_runner


@pytest.fixture
def test_input_node():
    CUSTOM_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    create_node_config(CUSTOM_CONFIG_DIR, PKD_NODE_NAME)

    config_node_input = {"input": ["none"], "output": ["test_output_1"]}

    return MockedNode(config_node_input)


@pytest.fixture
def test_node_end():
    CUSTOM_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    create_node_config(CUSTOM_CONFIG_DIR, PKD_NODE_NAME_2)

    config_node_end = {
        "input": ["test_output_1"],
        "output": ["test_output_2", "pipeline_end"],
    }

    return MockedNode(config_node_end)


@pytest.fixture
def runner_with_nodes(test_input_node, test_node_end):
    setup()
    instantiated_nodes = [test_input_node, test_node_end]
    test_runner = Runner(
        RUN_CONFIG_PATH, CONFIG_UPDATES_CLI, CUSTOM_DIR, instantiated_nodes
    )

    return test_runner


@pytest.mark.usefixtures("tmp_dir")
class TestRunner:
    def test_init_nodes_none(self, runner):
        assert isinstance(runner.pipeline, mock.Mock)
        assert runner.pipeline.nodes == []

    def test_init_nodes_with_instantiated_nodes(self, runner_with_nodes):
        with mock.patch(
            "peekingduck.pipeline.pipeline.Pipeline._check_pipe",
            wraps=replace_pipeline_check_pipe,
        ):
            assert runner_with_nodes.pipeline.nodes[0]._name == PKD_NODE
            assert runner_with_nodes.pipeline.nodes[0].inputs == ["none"]
            assert runner_with_nodes.pipeline.nodes[0].outputs == ["test_output_1"]

    def test_init_nodes_with_wrong_input(self):
        ground_truth = "pipeline"
        with mock.patch(
            "peekingduck.pipeline.pipeline.Pipeline.__init__", side_effect=ValueError
        ):
            with pytest.raises(SystemExit):
                Runner(RUN_CONFIG_PATH, CONFIG_UPDATES_CLI, CUSTOM_DIR, [ground_truth])

    def test_init_with_update_failure(self):
        setup()
        with mock.patch(
            "peekingduck.runner.check_requirements",
            side_effect=subprocess.CalledProcessError(1, ""),
        ), mock.patch(
            "peekingduck.declarative_loader.DeclarativeLoader.get_pipeline",
            wraps=get_pipeline_with_default_node_names,
        ):
            with pytest.raises(SystemExit):
                Runner(RUN_CONFIG_PATH, CONFIG_UPDATES_CLI, CUSTOM_DIR)

    def test_init_with_updated_packages(self):
        setup()
        with mock.patch(
            "peekingduck.runner.check_requirements", side_effect=[1]
        ), mock.patch(
            "peekingduck.declarative_loader.DeclarativeLoader.get_pipeline",
            wraps=get_pipeline_with_default_node_names,
        ):
            with pytest.raises(SystemExit):
                Runner(RUN_CONFIG_PATH, CONFIG_UPDATES_CLI, CUSTOM_DIR)

    def test_run(self, runner_with_nodes):
        with mock.patch(
            "peekingduck.runner.Runner.run",
            side_effect=Exception("End infinite while loop"),
        ):
            with pytest.raises(Exception):
                runner_with_nodes.pipeline.terminate = False
                runner_with_nodes.run()

        assert isinstance(runner_with_nodes.pipeline, object) == True

    def test_run_nodes(self, runner_with_nodes):
        correct_data = {
            "test_output_1": "test_output_0",
            "test_output_2": "test_output_0",
            "pipeline_end": "test_output_1",
        }
        runner_with_nodes.run()

        assert runner_with_nodes.pipeline.data == correct_data
        assert runner_with_nodes.pipeline.get_pipeline_results() == correct_data

    def test_pipeline_not_deleted_after_run(self, runner_with_nodes):
        assert isinstance(runner_with_nodes.pipeline, object) == True

        runner_with_nodes.pipeline.terminate = True
        runner_with_nodes.run()

        assert isinstance(runner_with_nodes.pipeline, object) == True

    def test_get_run_config(self, runner):
        node_list = runner.get_run_config()

        for idx, node in enumerate(node_list):
            assert node == NODES["nodes"][idx]
