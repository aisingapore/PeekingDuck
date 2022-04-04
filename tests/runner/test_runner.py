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

import sys
from pathlib import Path
from unittest import mock

import pytest
import yaml

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.runner import Runner
from peekingduck.utils.requirement_checker import RequirementChecker

PKD_NODE_TYPE = "pkd_node_type"
PKD_NODE_NAME = "pkd_node_name"
PKD_NODE_NAME_2 = "pkd_node_name2"
PKD_NODE = f"{PKD_NODE_TYPE}.{PKD_NODE_NAME}"
PKD_NODE_2 = f"{PKD_NODE_TYPE}.{PKD_NODE_NAME_2}"
NODES = {"nodes": [PKD_NODE, PKD_NODE_2]}

MODULE_DIR = Path("tmp_dir")
PIPELINE_PATH = MODULE_DIR / "pipeline_config.yml"
CUSTOM_NODES_DIR = MODULE_DIR / "custom_nodes"
CUSTOM_NODES_CONFIG_DIR = MODULE_DIR / "configs" / PKD_NODE_TYPE
PKD_NODE_DIR = MODULE_DIR / PKD_NODE_TYPE
CONFIG_UPDATES_CLI = "{'input.visual': {'resize':{'do_resizing':True}}}"


class MockedNode(AbstractNode):
    def __init__(self, config):
        super().__init__(config, node_path=PKD_NODE, pkd_base_dir=MODULE_DIR)

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


def create_pipeline_yaml(nodes):
    with open(PIPELINE_PATH, "w") as outfile:
        yaml.dump(nodes, outfile, default_flow_style=False)


def setup():
    sys.path.append(str(Path.cwd() / MODULE_DIR))
    PKD_NODE_DIR.mkdir(parents=True)
    create_pipeline_yaml(NODES)


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
def runner(request):
    setup()
    with mock.patch(
        "peekingduck.declarative_loader.DeclarativeLoader.get_pipeline",
        wraps=replace_declarativeloader_get_pipeline,
    ):
        test_runner = Runner(
            pipeline_path=PIPELINE_PATH,
            config_updates_cli=CONFIG_UPDATES_CLI,
            custom_nodes_parent_subdir=CUSTOM_NODES_DIR,
            nodes=request.param,
        )

        return test_runner


@pytest.fixture
def test_input_node():
    CUSTOM_NODES_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    create_node_config(CUSTOM_NODES_CONFIG_DIR, PKD_NODE_NAME)
    config_node_input = {"input": ["none"], "output": ["test_output_1"]}

    return MockedNode(config_node_input)


@pytest.fixture
def test_node_end():
    CUSTOM_NODES_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    create_node_config(CUSTOM_NODES_CONFIG_DIR, "pkd_node_name2")
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
        pipeline_path=PIPELINE_PATH,
        config_updates_cli=CONFIG_UPDATES_CLI,
        custom_nodes_parent_subdir=CUSTOM_NODES_DIR,
        nodes=instantiated_nodes,
    )

    return test_runner


@pytest.mark.usefixtures("tmp_dir")
class TestRunner:
    @pytest.mark.parametrize("runner", [None, []], indirect=True)
    def test_init_nodes_none(self, runner):
        assert isinstance(runner.pipeline, mock.Mock)
        assert runner.pipeline.nodes == []

    def test_init_nodes_none_config_updates_none(self, test_input_node):
        print(test_input_node)
        with pytest.raises(SystemExit):
            Runner(pipeline_path=PIPELINE_PATH)

    def test_init_nodes_none_custom_nodes_none(self):
        with pytest.raises(SystemExit):
            Runner(pipeline_path=PIPELINE_PATH, config_updates_cli=CONFIG_UPDATES_CLI)

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
        ), pytest.raises(SystemExit):
            Runner(
                pipeline_path=PIPELINE_PATH,
                config_updates_cli=CONFIG_UPDATES_CLI,
                custom_nodes_parent_subdir=CUSTOM_NODES_DIR,
                nodes=[ground_truth],
            )

    def test_init_with_updated_packages(self):
        setup()
        with mock.patch.object(RequirementChecker, "n_update", 1), mock.patch(
            "peekingduck.declarative_loader.DeclarativeLoader.get_pipeline",
            wraps=get_pipeline_with_default_node_names,
        ), pytest.raises(SystemExit) as exec_info:
            Runner(
                pipeline_path=PIPELINE_PATH,
                config_updates_cli=CONFIG_UPDATES_CLI,
                custom_nodes_parent_subdir=CUSTOM_NODES_DIR,
            )
        # Ensure we are throwing the correct exit code
        assert exec_info.value.code == 3

    def test_run(self, runner_with_nodes):
        with mock.patch(
            "peekingduck.runner.Runner.run",
            side_effect=Exception("End infinite while loop"),
        ), pytest.raises(Exception):
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

    @pytest.mark.parametrize("runner", [None], indirect=True)
    def test_get_pipeline(self, runner):
        node_list = runner.get_pipeline()

        for idx, (node, _) in enumerate(node_list):
            assert node == NODES["nodes"][idx]
