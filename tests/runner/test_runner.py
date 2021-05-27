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
import importlib
import textwrap
from unittest import mock

import yaml
import pytest
from peekingduck.runner import Runner

PKD_NODE_TYPE = "pkd_node_type"
PKD_NODE_NAME = "pkd_node_name"
PKD_NODE = "pkd_node_type" + "." + "pkd_node_name"
NODES = {"nodes": [PKD_NODE]}

MODULE_PATH = "tmp_dir"
RUN_CONFIG_PATH = os.path.join(MODULE_PATH, "run_config.yml")
CUSTOM_FOLDER_PATH = os.path.join(MODULE_PATH, "custom_nodes")
PKD_NODE_DIR = os.path.join(MODULE_PATH, PKD_NODE_TYPE)
PKD_NODE_CONFIG_DIR = os.path.join(MODULE_PATH, "configs", PKD_NODE_TYPE)


def create_run_config_yaml(nodes):

    with open(RUN_CONFIG_PATH, 'w') as outfile:
        yaml.dump(nodes, outfile, default_flow_style=False)


def create_node_python(node_dir, node_name):

    node_file = node_name + ".py"
    with open(os.path.join(node_dir,  node_file), 'w') as fp:
        content = textwrap.dedent(
            """\
            from peekingduck.pipeline.nodes.node import AbstractNode
            class Node(AbstractNode):
                def __init__(self, config):
                    super().__init__(config, node_path=__name__)
                def run(self):
                    return {}
            """)

        fp.write(content)


def create_node_config(config_dir, node_name):

    config_text = {"root": None,
                   "input": ["test_source"],
                   "output": ["test_end"]}

    node_config_file = node_name + ".yml"
    with open(os.path.join(config_dir,  node_config_file), 'w') as fp:
        yaml.dump(config_text, fp)


def setup():
    sys.path.append(MODULE_PATH)

    os.makedirs(PKD_NODE_DIR)
    os.makedirs(PKD_NODE_CONFIG_DIR)

    create_run_config_yaml(NODES)

    create_node_python(PKD_NODE_DIR, PKD_NODE_NAME)
    create_node_config(PKD_NODE_CONFIG_DIR, PKD_NODE_NAME)


def instantiate_nodes():
    instantiated_nodes = []

    node_path = PKD_NODE_TYPE + "." + PKD_NODE_NAME
    node_config_path = os.path.join(MODULE_PATH, "configs",
                                    PKD_NODE_TYPE, PKD_NODE_NAME + ".yml")

    node = importlib.import_module(node_path)
    with open(node_config_path) as file:
        node_config = yaml.load(file, Loader=yaml.FullLoader)

    instantiated_nodes.append(node.Node(node_config))

    return instantiated_nodes


def replace_declarativeloader_get_nodes():
    return True


def replace_pipeline_check_pipe(node):
    pass


@ pytest.fixture
def runner():

    setup()

    with mock.patch('peekingduck.loaders.DeclarativeLoader.get_nodes',
                    wraps=replace_declarativeloader_get_nodes):

        test_runner = Runner(RUN_CONFIG_PATH,
                             CUSTOM_FOLDER_PATH)

        return test_runner


@ pytest.fixture
def runner_with_nodes():

    setup()

    instantiated_nodes = instantiate_nodes()

    test_runner = Runner(RUN_CONFIG_PATH,
                         CUSTOM_FOLDER_PATH,
                         instantiated_nodes)

    return test_runner


@pytest.mark.usefixtures("tmp_dir")
class TestRunner:

    def test_init_nodes_none(self, runner):

        assert runner.pipeline == True

    def test_init_nodes_empty(self, runner_with_nodes):

        with mock.patch('peekingduck.pipeline.pipeline.Pipeline._check_pipe',
                        wraps=replace_pipeline_check_pipe):

            assert runner_with_nodes.pipeline.nodes[0]._name == PKD_NODE
            assert runner_with_nodes.pipeline.nodes[0]._inputs == ["test_source"]
            assert runner_with_nodes.pipeline.nodes[0]._outputs == ["test_end"]

    def test_init_nodes(self):

        ground_truth = "pipeline"

        with pytest.raises(AttributeError):

            Runner(RUN_CONFIG_PATH, CUSTOM_FOLDER_PATH, [ground_truth])

    def test_run(self, runner_with_nodes):

        with mock.patch('peekingduck.pipeline.pipeline.Pipeline.execute',
                        side_effect=Exception("End infinite while loop")):

            with pytest.raises(Exception):

                runner_with_nodes.pipeline.video_end = False
                runner_with_nodes.run()

                assert isinstance(runner_with_nodes.pipeline, object) == True

    def test_run_delete(self, runner_with_nodes):

        assert isinstance(runner_with_nodes.pipeline, object) == True

        runner_with_nodes.pipeline.video_end = True

        runner_with_nodes.run()

        with pytest.raises(AttributeError):
            assert isinstance(runner_with_nodes.pipeline, object) == True

    def test_get_run_config(self, runner):

        node_list = runner.get_run_config()

        for idx, node in enumerate(node_list):
            assert node == NODES["nodes"][idx]
