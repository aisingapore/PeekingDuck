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
import sys
from unittest import mock
import unittest

import yaml
import pytest
from peekingduck.runner import Runner
from peekingduck.pipeline.nodes.node import AbstractNode

PKD_NODE_TYPE = "pkd_node_type"
PKD_NODE_NAME = "pkd_node_name"
PKD_NODE = "pkd_node_type" + "." + "pkd_node_name"
NODES = {"nodes": [PKD_NODE]}

MODULE_PATH = "tmp_dir"
RUN_CONFIG_PATH = os.path.join(MODULE_PATH, "run_config.yml")
CUSTOM_FOLDER_PATH = os.path.join(MODULE_PATH, "custom_nodes")
PKD_NODE_DIR = os.path.join(MODULE_PATH, PKD_NODE_TYPE)
CONFIG_UPDATES_CLI = "{'input.live': {'resize':{'do_resizing':True}}}"


class MockedNode(AbstractNode):
    def __init__(self, config):
        super().__init__(config, node_path=PKD_NODE)

    def run(self, inputs):
        output = {}
        for idx in range(len(self._outputs)):
            output[self._outputs[idx]] = "test_output_" + str(idx)

        return output


def create_run_config_yaml(nodes):

    with open(RUN_CONFIG_PATH, 'w') as outfile:
        yaml.dump(nodes, outfile, default_flow_style=False)


def setup():
    sys.path.append(MODULE_PATH)

    os.makedirs(PKD_NODE_DIR)

    create_run_config_yaml(NODES)


def replace_declarativeloader_get_pipeline():
    return True


def replace_pipeline_check_pipe(node):
    pass


def replace_pipeline(node):
    pass


@ pytest.fixture
def runner():

    setup()

    with mock.patch('peekingduck.loaders.DeclarativeLoader.get_pipeline',
                    wraps=replace_declarativeloader_get_pipeline):

        test_runner = Runner(RUN_CONFIG_PATH,
                             CONFIG_UPDATES_CLI,
                             CUSTOM_FOLDER_PATH)

        return test_runner


@pytest.fixture
def test_input_node():

    config_node_input = {'input': ["source"],
                         'output': ["test_output_1"]}

    return MockedNode(config_node_input)


@ pytest.fixture
def runner_with_nodes(test_input_node):

    setup()

    instantiated_nodes = [test_input_node]

    test_runner = Runner(RUN_CONFIG_PATH,
                         CONFIG_UPDATES_CLI,
                         CUSTOM_FOLDER_PATH,
                         instantiated_nodes)

    return test_runner


@pytest.mark.usefixtures("tmp_dir")
class TestRunner:

    def test_init_nodes_none(self, runner):

        assert runner.pipeline == True

    def test_init_nodes_with_instantiated_nodes(self, runner_with_nodes):

        with mock.patch('peekingduck.pipeline.pipeline.Pipeline._check_pipe',
                        wraps=replace_pipeline_check_pipe):

            assert runner_with_nodes.pipeline.nodes[0]._name == PKD_NODE
            assert runner_with_nodes.pipeline.nodes[0]._inputs == ["source"]
            assert runner_with_nodes.pipeline.nodes[0]._outputs == [
                "test_output_1"]

    def test_init_nodes_with_wrong_input(self):

        ground_truth = "pipeline"

        with mock.patch('peekingduck.pipeline.pipeline.Pipeline.__init__',
                        side_effect=ValueError):

            with pytest.raises(SystemExit):
                Runner(RUN_CONFIG_PATH, CONFIG_UPDATES_CLI,
                       CUSTOM_FOLDER_PATH, [ground_truth])

    def test_run(self, runner_with_nodes):

        with mock.patch('peekingduck.pipeline.pipeline.Pipeline.execute',
                        side_effect=Exception("End infinite while loop")):

            with pytest.raises(Exception):

                runner_with_nodes.pipeline.terminate = False
                runner_with_nodes.run()

        assert isinstance(runner_with_nodes.pipeline, object) == True

    def test_run_delete(self, runner_with_nodes):

        assert isinstance(runner_with_nodes.pipeline, object) == True

        runner_with_nodes.pipeline.terminate = True

        runner_with_nodes.run()

        with pytest.raises(AttributeError):
            assert isinstance(runner_with_nodes.pipeline, object) == True

    def test_get_run_config(self, runner):

        node_list = runner.get_run_config()

        for idx, node in enumerate(node_list):
            assert node == NODES["nodes"][idx]
