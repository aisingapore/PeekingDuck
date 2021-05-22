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
from unittest import mock
from peekingduck.loaders import DeclarativeLoader

dir_path = "tmp_dir"
RUN_PATH = os.path.join(dir_path, "run_config.yml")
CUSTOM_NODE_PATH = os.path.join(dir_path, "custom_nodes")
NODE_TYPE = "test_type"
NODE_NAME = "test_name"
CUSTOM_NODE = 'custom_nodes.'
CUSTOM_NODE_NAME = 'output.test'
NODES = {"nodes": [NODE_TYPE + "." + NODE_NAME,
                   CUSTOM_NODE + CUSTOM_NODE_NAME]}


def create_run_config_yaml(nodes):

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(RUN_PATH, 'w') as outfile:
        yaml.dump(nodes, outfile, default_flow_style=False)


def create_node_python():

    node_folder_path = os.path.join(dir_path, NODE_TYPE)
    if not os.path.exists(node_folder_path):
        os.makedirs(node_folder_path)

    node_file = NODE_NAME + ".py"
    with open(os.path.join(node_folder_path,  node_file), 'w') as fp:
        fp.write("from peekingduck.pipeline.nodes.node import AbstractNode\
                 \nclass Node(AbstractNode):\
                 \n def __init__(self, config):\
                 \n  super().__init__(config, node_path=__name__)\
                 \n def run(self):\
                 \n  return {}")


def create_node_config():
    node_config_folder = os.path.join(dir_path, 'configs', NODE_TYPE)
    if not os.path.exists(node_config_folder):
        os.makedirs(node_config_folder)

    config_text = {"root": None,
                   "input": False,
                   "output": False}

    node_config_file = NODE_NAME + ".yml"
    with open(os.path.join(node_config_folder,  node_config_file), 'w') as fp:
        yaml.dump(config_text, fp)


@ pytest.fixture
def declarativeloader():

    sys.path.append(dir_path)

    create_run_config_yaml(NODES)

    config_loader = DeclarativeLoader(RUN_PATH, CUSTOM_NODE_PATH)

    return config_loader


def replace_init_node(path_to_node, node_name, config_loader, config_updates):
    return [path_to_node, node_name, config_loader, config_updates]


def replace_instantiate_nodes():
    return None


@ pytest.mark.usefixtures("tmp_dir")
class TestDeclarativeLoader:

    def test_load_node_list(self, declarativeloader):

        loaded_nodes = declarativeloader._load_node_list(RUN_PATH)

        for idx, node in enumerate(loaded_nodes):
            assert node == NODES["nodes"][idx]

    def test_instantiate_nodes(self, declarativeloader):

        node_one = ['peekingduck.pipeline.nodes.',
                    NODE_TYPE + "." + NODE_NAME,
                    declarativeloader.config_loader,
                    None]

        node_two = [CUSTOM_NODE,
                    CUSTOM_NODE_NAME,
                    declarativeloader.custom_config_loader,
                    None]

        ground_truth = [node_one, node_two]

        with mock.patch('peekingduck.loaders.DeclarativeLoader._init_node',
                        wraps=replace_init_node):

            instantiated_nodes = declarativeloader._instantiate_nodes()

            for node_num, node in enumerate(instantiated_nodes):
                for idx, output in enumerate(node):
                    assert output == ground_truth[node_num][idx]

    def test_init_node(self, declarativeloader):
        create_node_python()
        create_node_config()

        declarativeloader.config_loader._basedir = dir_path

        path_to_node = ""
        node_name = NODE_TYPE + "." + NODE_NAME
        config_loader = declarativeloader.config_loader
        config_updates = None

        init_node = declarativeloader._init_node(path_to_node,
                                                 node_name,
                                                 config_loader,
                                                 config_updates)

        assert init_node._name == node_name
        assert init_node._inputs == False
        assert init_node._outputs == False

    def test_init_node_edit(self, declarativeloader):
        create_node_python()
        create_node_config()

        declarativeloader.config_loader._basedir = dir_path

        path_to_node = ""
        node_name = NODE_TYPE + "." + NODE_NAME
        config_loader = declarativeloader.config_loader
        config_updates = {"input": True}

        init_node = declarativeloader._init_node(path_to_node,
                                                 node_name,
                                                 config_loader,
                                                 config_updates)

        assert init_node._name == node_name
        assert init_node._inputs == True
        assert init_node._outputs == False

    def test_edit_node_config(self, declarativeloader):

        config = {'input': False,
                  'output': True}
        config_update = {'input': True}
        ground_truth = {'input': True,
                        'output': True}

        config = declarativeloader._edit_node_config(config, config_update)

        for key in config.keys():
            assert config[key] == ground_truth[key]

    def test_get_nodes(self, declarativeloader):

        with pytest.raises(TypeError):
            with mock.patch('peekingduck.loaders.DeclarativeLoader._instantiate_nodes',
                            wraps=replace_instantiate_nodes):

                declarativeloader.get_nodes()
