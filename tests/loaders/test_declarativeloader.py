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
import string
import random
import importlib
import textwrap
from unittest import mock

import yaml
import pytest
from peekingduck.declarative_loader import DeclarativeLoader

PKD_NODE_TYPE = "input"
PKD_NODE_NAME = "pkd_node_name"
PKD_NODE = f"{PKD_NODE_TYPE}.{PKD_NODE_NAME}"
CUSTOM_NODE_TYPE = "custom_node_type"
CUSTOM_NODE_NAME = "custom_node_name"
CUSTOM_NODE = f"{CUSTOM_NODE_TYPE}.{CUSTOM_NODE_NAME}"
NODES = {"nodes": [PKD_NODE,
                   {PKD_NODE: [{'setting': True}]},
                   f"{CUSTOM_NODE_NAME}.{CUSTOM_NODE}"]}

MODULE_PATH = "tmp_dir"
UNIQUE_SUFFIX = ''.join(random.choice(string.ascii_lowercase)
                        for x in range(8))
CUSTOM_FOLDER_NAME = f"custom_nodes_{UNIQUE_SUFFIX}"
RUN_CONFIG_PATH = os.path.join(MODULE_PATH, "run_config.yml")
CUSTOM_FOLDER_PATH = os.path.join(MODULE_PATH, CUSTOM_FOLDER_NAME)
PKD_NODE_DIR = os.path.join(MODULE_PATH, PKD_NODE_TYPE)
CUSTOM_NODE_DIR = os.path.join(CUSTOM_FOLDER_PATH, CUSTOM_NODE_TYPE)
PKD_NODE_CONFIG_DIR = os.path.join(MODULE_PATH, "configs", PKD_NODE_TYPE)
CUSTOM_NODE_CONFIG_DIR = os.path.join(
    CUSTOM_FOLDER_PATH, "configs", CUSTOM_NODE_TYPE)
CONFIG_UPDATES_CLI = "{'input.live': {'resize':{'do_resizing':True, 'width':320}}}"


def create_run_config_yaml(nodes):

    with open(RUN_CONFIG_PATH, 'w') as outfile:
        yaml.dump(nodes, outfile, default_flow_style=False)


def create_node_python(node_dir, node_name):

    node_file = f"{node_name}.py"
    with open(os.path.join(node_dir, node_file), 'w') as fp:
        content = textwrap.dedent(
            """\
            import pathlib
            from peekingduck.pipeline.nodes.node import AbstractNode
            
            class Node(AbstractNode):
                def __init__(self, config, pkdbasedir=None):
                    pkdbasedir = str(pathlib.Path(__file__).parents[1].resolve())
                    super().__init__(config, node_path=__name__, pkdbasedir=pkdbasedir)

                def run(self):
                    return {}
            """)

        fp.write(content)


def create_node_config(config_dir, node_name):

    config_text = {"root": None,
                   "input": ["source"],
                   "output": ["end"]}

    node_config_file = f"{node_name}.yml"

    with open(os.path.join(config_dir, node_config_file), 'w') as fp:
        yaml.dump(config_text, fp)


def setup():
    sys.path.append(MODULE_PATH)

    os.makedirs(PKD_NODE_DIR)
    os.makedirs(CUSTOM_NODE_DIR)
    os.makedirs(PKD_NODE_CONFIG_DIR)
    os.makedirs(CUSTOM_NODE_CONFIG_DIR)

    create_run_config_yaml(NODES)

    create_node_python(PKD_NODE_DIR, PKD_NODE_NAME)
    create_node_python(CUSTOM_NODE_DIR, CUSTOM_NODE_NAME)

    create_node_config(PKD_NODE_CONFIG_DIR, PKD_NODE_NAME)
    create_node_config(CUSTOM_NODE_CONFIG_DIR, CUSTOM_NODE_NAME)


@ pytest.fixture
def declarativeloader():

    setup()

    declarative_loader = DeclarativeLoader(
        RUN_CONFIG_PATH, CONFIG_UPDATES_CLI, MODULE_PATH)

    declarative_loader.config_loader._basedir = MODULE_PATH
    declarative_loader.custom_config_loader._basedir = CUSTOM_FOLDER_PATH

    return declarative_loader


def replace_init_node(path_to_node, node_name, config_loader, config_updates):
    return [path_to_node, node_name, config_loader, config_updates]


def replace_instantiate_nodes_return_none():
    return None


def replace_instantiate_nodes():
    instantiated_nodes = []

    node_path = PKD_NODE
    node_config_path = os.path.join(PKD_NODE_CONFIG_DIR,
                                    f"{PKD_NODE_NAME}.yml")

    node = importlib.import_module(node_path)
    with open(node_config_path) as file:
        node_config = yaml.safe_load(file)

    instantiated_nodes.append(node.Node(node_config, pkdbasedir=MODULE_PATH))

    return instantiated_nodes


@ pytest.mark.usefixtures("tmp_dir")
class TestDeclarativeLoader:
    # NOTE Due to essential code logic, private functions are separately tested
    # from public function to ensure simple, yet robust and comprehensive unit
    # testings.

    def test_loaded_node_list(self, declarativeloader):

        loaded_nodes = declarativeloader.node_list

        for idx, node in enumerate(loaded_nodes):
            assert node == NODES["nodes"][idx]

    def test_get_custom_name_from_node_list(self, declarativeloader):

        custom_folder_name = declarativeloader._get_custom_name_from_node_list()

        assert custom_folder_name == CUSTOM_NODE_NAME

    def test_instantiate_nodes(self, declarativeloader):

        pkd_node_default = ['peekingduck.pipeline.nodes.',
                            PKD_NODE,
                            declarativeloader.config_loader,
                            None]

        pkd_node_edit = ['peekingduck.pipeline.nodes.',
                         PKD_NODE,
                         declarativeloader.config_loader,
                         [{'setting': True}]]

        custom_node = [f"{CUSTOM_NODE_NAME}.",
                       CUSTOM_NODE,
                       declarativeloader.custom_config_loader,
                       None]

        ground_truth = [pkd_node_default, pkd_node_edit, custom_node]

        with mock.patch('peekingduck.declarative_loader.DeclarativeLoader._init_node',
                        wraps=replace_init_node):

            instantiated_nodes = declarativeloader._instantiate_nodes()

            for node_num, node in enumerate(instantiated_nodes):
                for idx, output in enumerate(node):
                    assert output == ground_truth[node_num][idx]

    def test_init_node_pkd(self, declarativeloader):

        path_to_node = ""
        node_name = PKD_NODE
        config_loader = declarativeloader.config_loader
        config_updates = None

        init_node = declarativeloader._init_node(path_to_node,
                                                 node_name,
                                                 config_loader,
                                                 config_updates)

        assert init_node._name == node_name
        assert init_node.inputs == ['source']
        assert init_node.outputs == ['end']

    def test_init_node_custom(self, declarativeloader):

        path_to_node = f"{CUSTOM_FOLDER_NAME}."
        node_name = CUSTOM_NODE
        config_loader = declarativeloader.custom_config_loader
        config_updates = None

        init_node = declarativeloader._init_node(path_to_node,
                                                 node_name,
                                                 config_loader,
                                                 config_updates)

        assert init_node._name == f"{path_to_node}{node_name}"
        assert init_node.inputs == ['source']
        assert init_node.outputs == ['end']

    def test_init_node_edit(self, declarativeloader):

        path_to_node = ""
        node_name = PKD_NODE
        config_loader = declarativeloader.config_loader
        config_updates = {"input": ['img']}

        init_node = declarativeloader._init_node(path_to_node,
                                                 node_name,
                                                 config_loader,
                                                 config_updates)

        assert init_node._name == node_name
        assert init_node.inputs == ['img']
        assert init_node.outputs == ['end']

    def test_edit_config(self, declarativeloader):

        node_name = 'input.live'
        orig_config = {
            'mirror_image': True,
            'resize': {
                'do_resizing': False,
                'width': 1280,
                'height': 720
            }}
        config_update = {'mirror_image': False,
                         'resize': {'do_resizing': True},
                         'invalid_key': 123}
        ground_truth = {
            'mirror_image': False,
            'resize': {
                'do_resizing': True,
                'width': 1280,
                'height': 720
            }}

        orig_config = declarativeloader._edit_config(
            orig_config, config_update, node_name)

        assert orig_config['mirror_image'] == ground_truth['mirror_image']
        assert orig_config['resize']['do_resizing'] == ground_truth['resize']['do_resizing']
        # Ensure that config_update does not replace the "resize" sub-dict completely and
        # erase "width" or "height"
        assert "width" in orig_config["resize"]
        assert "invalid_key" not in ground_truth

    def test_get_pipeline(self, declarativeloader):

        with mock.patch('peekingduck.declarative_loader.DeclarativeLoader._instantiate_nodes',
                        wraps=replace_instantiate_nodes):

            pipeline = declarativeloader.get_pipeline()
            assert pipeline.nodes[0]._name == PKD_NODE
            assert pipeline.nodes[0].inputs == ['source']
            assert pipeline.nodes[0].outputs == ['end']

    def test_get_pipeline_error(self, declarativeloader):

        with pytest.raises(TypeError):
            with mock.patch('peekingduck.declarative_loader.DeclarativeLoader._instantiate_nodes',
                            wraps=replace_instantiate_nodes_return_none):

                declarativeloader.get_pipeline()
