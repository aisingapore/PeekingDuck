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
import yaml

import pytest
import importlib
from unittest import mock
from peekingduck.runner import Runner

TEMP_BASE_PATH = "tmp_dir"
RUN_CONFIG_PATH = os.path.join(TEMP_BASE_PATH, "run_config.yml")
CUSTOM_FOLDER_PATH = os.path.join(TEMP_BASE_PATH, "custom_nodes")

PKD_NODE_TYPE = "pkd_node_type"
PKD_NODE_NAME = "pkd_node_name"
CUSTOM_NODE_TYPE = "custom_node_type"
CUSTOM_NODE_NAME = "custom_node_name"
NODES = {"nodes": [PKD_NODE_TYPE + "." + PKD_NODE_NAME,
                   "custom_nodes." + CUSTOM_NODE_TYPE + "." + CUSTOM_NODE_NAME]}


def create_run_config_yaml(nodes):

    if not os.path.exists(TEMP_BASE_PATH):
        os.makedirs(TEMP_BASE_PATH)

    with open(RUN_CONFIG_PATH, 'w') as outfile:
        yaml.dump(nodes, outfile, default_flow_style=False)


def create_node_python(folder, node_type, node_name):

    node_folder_path = os.path.join(folder, node_type)
    if not os.path.exists(node_folder_path):
        os.makedirs(node_folder_path)

    node_file = node_name + ".py"
    with open(os.path.join(node_folder_path,  node_file), 'w') as fp:
        fp.write("from peekingduck.pipeline.nodes.node import AbstractNode\
                 \nclass Node(AbstractNode):\
                 \n def __init__(self, config):\
                 \n  super().__init__(config, node_path=__name__)\
                 \n def run(self, inputs):\
                 \n  return {}")


def create_node_config(folder, node_type, node_name):
    node_config_folder = os.path.join(folder, 'configs', node_type)
    if not os.path.exists(node_config_folder):
        os.makedirs(node_config_folder)

    config_text = {"root": None,
                   "input": ["source", "end"],
                   "output": ["end"]}

    node_config_file = node_name + ".yml"
    with open(os.path.join(node_config_folder,  node_config_file), 'w') as fp:
        yaml.dump(config_text, fp)


@pytest.fixture
def setup():
    sys.path.append(TEMP_BASE_PATH)

    create_run_config_yaml(NODES)

    create_node_python(TEMP_BASE_PATH, PKD_NODE_TYPE, PKD_NODE_NAME)
    create_node_python(CUSTOM_FOLDER_PATH, CUSTOM_NODE_TYPE, CUSTOM_NODE_NAME)

    create_node_config(TEMP_BASE_PATH, PKD_NODE_TYPE, PKD_NODE_NAME)
    create_node_config(CUSTOM_FOLDER_PATH, CUSTOM_NODE_TYPE, CUSTOM_NODE_NAME)


def replace_declarativeloader_get_nodes():
    return True


def replace_pipeline_check_pipe(node):
    pass


def replace_pipeline_execute():
    pass
    # return "Ran pipeline execute"


def instantiate_nodes():
    instantiated_nodes = []

    node_path = PKD_NODE_TYPE + "." + PKD_NODE_NAME
    node_config_path = os.path.join(TEMP_BASE_PATH, "configs",
                                    PKD_NODE_TYPE, PKD_NODE_NAME + ".yml")

    node = importlib.import_module(node_path)
    with open(node_config_path) as file:
        node_config = yaml.load(file, Loader=yaml.FullLoader)

    instantiated_nodes.append(node.Node(node_config))

    return instantiated_nodes


@pytest.mark.usefixtures("tmp_dir")
class TestRunner:

    def test_init_nodes_none(self, setup):

        with mock.patch('peekingduck.loaders.DeclarativeLoader.get_nodes',
                        wraps=replace_declarativeloader_get_nodes):

            test_runner = Runner(RUN_CONFIG_PATH,
                                 CUSTOM_FOLDER_PATH)

            assert test_runner.pipeline == True

    def test_init_nodes_empty(self, setup):

        ground_truth = "pipeline"

        with mock.patch('peekingduck.pipeline.pipeline.Pipeline._check_pipe',
                        wraps=replace_pipeline_check_pipe):

            test_runner = Runner(RUN_CONFIG_PATH,
                                 CUSTOM_FOLDER_PATH,
                                 [ground_truth])

            assert test_runner.pipeline.nodes[0] == ground_truth

    def test_init_nodes(self, setup):

        ground_truth = "pipeline"

        with pytest.raises(AttributeError):

            Runner(RUN_CONFIG_PATH, CUSTOM_FOLDER_PATH, [ground_truth])

    def test_run_delete(self, setup):

        instantiated_nodes = instantiate_nodes()

        with mock.patch('peekingduck.pipeline.pipeline.Pipeline.execute',
                        wraps=replace_pipeline_execute):

            test_runner = Runner(RUN_CONFIG_PATH,
                                 CUSTOM_FOLDER_PATH,
                                 instantiated_nodes)

            assert isinstance(test_runner.pipeline, object) == True

            test_runner.pipeline.video_end = True

            test_runner.run()

            with pytest.raises(AttributeError):
                assert isinstance(test_runner.pipeline, object) == True

    def test_run_delete(self, setup):

        instantiated_nodes = instantiate_nodes()

        with mock.patch('peekingduck.pipeline.pipeline.Pipeline.execute',
                        wraps=replace_pipeline_execute):

            test_runner = Runner(RUN_CONFIG_PATH,
                                 CUSTOM_FOLDER_PATH,
                                 instantiated_nodes)

            assert isinstance(test_runner.pipeline, object) == True

            test_runner.pipeline.video_end = True

            test_runner.run()

            with pytest.raises(AttributeError):
                assert isinstance(test_runner.pipeline, object) == True

    def test_get_run_config(self, setup):

        with mock.patch('peekingduck.loaders.DeclarativeLoader.get_nodes',
                        wraps=replace_declarativeloader_get_nodes):

            test_runner = Runner(RUN_CONFIG_PATH,
                                 CUSTOM_FOLDER_PATH)

            node_list = test_runner.get_run_config()

            for idx, node in enumerate(node_list):
                assert node == NODES["nodes"][idx]
