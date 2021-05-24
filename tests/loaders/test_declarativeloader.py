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
                 \n def run(self):\
                 \n  return {}")


def create_node_config(folder, node_type, node_name):
    node_config_folder = os.path.join(folder, 'configs', node_type)
    if not os.path.exists(node_config_folder):
        os.makedirs(node_config_folder)

    config_text = {"root": None,
                   "input": False,
                   "output": False}

    node_config_file = node_name + ".yml"
    with open(os.path.join(node_config_folder,  node_config_file), 'w') as fp:
        yaml.dump(config_text, fp)


def setup():
    sys.path.append(TEMP_BASE_PATH)

    create_run_config_yaml(NODES)

    create_node_python(TEMP_BASE_PATH, PKD_NODE_TYPE, PKD_NODE_NAME)
    create_node_python(CUSTOM_FOLDER_PATH, CUSTOM_NODE_TYPE, CUSTOM_NODE_NAME)

    create_node_config(TEMP_BASE_PATH, PKD_NODE_TYPE, PKD_NODE_NAME)
    create_node_config(CUSTOM_FOLDER_PATH, CUSTOM_NODE_TYPE, CUSTOM_NODE_NAME)


@ pytest.fixture
def declarativeloader():

    setup()

    config_loader = DeclarativeLoader(RUN_CONFIG_PATH, CUSTOM_FOLDER_PATH)

    return config_loader


def replace_init_node(path_to_node, node_name, config_loader, config_updates):
    return [path_to_node, node_name, config_loader, config_updates]


def replace_instantiate_nodes():
    return None


@ pytest.mark.usefixtures("tmp_dir")
class TestDeclarativeLoader:

    def test_load_node_list(self, declarativeloader):

        loaded_nodes = declarativeloader._load_node_list(RUN_CONFIG_PATH)

        for idx, node in enumerate(loaded_nodes):
            assert node == NODES["nodes"][idx]

    def test_instantiate_nodes(self, declarativeloader):

        pkd_node = ['peekingduck.pipeline.nodes.',
                    PKD_NODE_TYPE + "." + PKD_NODE_NAME,
                    declarativeloader.config_loader,
                    None]

        custom_node = ["custom_nodes.",
                       CUSTOM_NODE_TYPE + "." + CUSTOM_NODE_NAME,
                       declarativeloader.custom_config_loader,
                       None]

        ground_truth = [pkd_node, custom_node]

        with mock.patch('peekingduck.loaders.DeclarativeLoader._init_node',
                        wraps=replace_init_node):

            instantiated_nodes = declarativeloader._instantiate_nodes()

            for node_num, node in enumerate(instantiated_nodes):
                for idx, output in enumerate(node):
                    assert output == ground_truth[node_num][idx]

    def test_init_node_pkd(self, declarativeloader):

        declarativeloader.config_loader._basedir = TEMP_BASE_PATH

        path_to_node = ""
        node_name = PKD_NODE_TYPE + "." + PKD_NODE_NAME
        config_loader = declarativeloader.config_loader
        config_updates = None

        init_node = declarativeloader._init_node(path_to_node,
                                                 node_name,
                                                 config_loader,
                                                 config_updates)

        assert init_node._name == node_name
        assert init_node._inputs == False
        assert init_node._outputs == False

    def test_init_node_custom(self, declarativeloader):

        declarativeloader.custom_config_loader._basedir = CUSTOM_FOLDER_PATH

        path_to_node = "custom_nodes."
        node_name = CUSTOM_NODE_TYPE + "." + CUSTOM_NODE_NAME
        config_loader = declarativeloader.custom_config_loader
        config_updates = None

        init_node = declarativeloader._init_node(path_to_node,
                                                 node_name,
                                                 config_loader,
                                                 config_updates)

        assert init_node._name == path_to_node + node_name
        assert init_node._inputs == False
        assert init_node._outputs == False

    def test_init_node_edit(self, declarativeloader):

        declarativeloader.config_loader._basedir = TEMP_BASE_PATH

        path_to_node = ""
        node_name = PKD_NODE_TYPE + "." + PKD_NODE_NAME
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

    def test_get_nodes_raise_error(self, declarativeloader):

        with pytest.raises(TypeError):
            with mock.patch('peekingduck.loaders.DeclarativeLoader._instantiate_nodes',
                            wraps=replace_instantiate_nodes):

                declarativeloader.get_nodes()
