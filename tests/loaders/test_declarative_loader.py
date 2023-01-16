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

import importlib
import random
import string
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import yaml

from peekingduck.declarative_loader import DeclarativeLoader

PKD_NODE_TYPE = "input"
PKD_NODE_NAME = "pkd_node_name"
PKD_NODE = f"{PKD_NODE_TYPE}.{PKD_NODE_NAME}"
CUSTOM_NODE_TYPE = "custom_node_type"
CUSTOM_NODE_NAME = "custom_node_name"
CUSTOM_NODE = f"{CUSTOM_NODE_TYPE}.{CUSTOM_NODE_NAME}"
NODES = {
    "nodes": [
        PKD_NODE,
        {PKD_NODE: [{"setting": True}]},
        f"{CUSTOM_NODE_NAME}.{CUSTOM_NODE}",
    ]
}

UNIQUE_SUFFIX = "".join(random.choice(string.ascii_lowercase) for _ in range(8))
CUSTOM_NODES_DIR_NAME = f"custom_nodes_{UNIQUE_SUFFIX}"

CONFIG_UPDATES_CLI = "{'input.visual': {'resize':{'do_resizing':True, 'width':320}}}"


@pytest.fixture(name="declarative_loader")
def fixture_declarative_loader():
    paths = get_paths(Path.cwd())
    prepare_environment(paths)

    declarative_loader = DeclarativeLoader(
        paths.pipeline_path, CONFIG_UPDATES_CLI, paths.module_dir
    )
    declarative_loader.config_loader._base_dir = paths.module_dir
    declarative_loader.custom_config_loader._base_dir = paths.custom_nodes_dir

    return declarative_loader


def get_paths(root_dir):
    module_dir = root_dir / "tmp_dir"
    pipeline_path = module_dir / "pipeline_config.yml"
    custom_nodes_dir = module_dir / CUSTOM_NODES_DIR_NAME
    pkd_node_dir = module_dir / PKD_NODE_TYPE
    custom_node_dir = custom_nodes_dir / CUSTOM_NODE_TYPE
    pkd_node_config_dir = module_dir / "configs" / PKD_NODE_TYPE
    custom_node_config_dir = custom_nodes_dir / "configs" / CUSTOM_NODE_TYPE
    return SimpleNamespace(
        module_dir=module_dir,
        pipeline_path=pipeline_path,
        custom_nodes_dir=custom_nodes_dir,
        pkd_node_dir=pkd_node_dir,
        custom_node_dir=custom_node_dir,
        pkd_node_config_dir=pkd_node_config_dir,
        custom_node_config_dir=custom_node_config_dir,
    )


def create_pipeline_yaml(pipeline_path, nodes):
    with open(pipeline_path, "w") as outfile:
        yaml.dump(nodes, outfile, default_flow_style=False)


def create_node_python(node_dir, node_name):
    node_file = f"{node_name}.py"
    with open(node_dir / node_file, "w") as fp:
        content = textwrap.dedent(
            """\
            import pathlib
            from peekingduck.nodes.abstract_node import AbstractNode

            class Node(AbstractNode):
                def __init__(self, config, pkd_base_dir=None):
                    pkd_base_dir = str(pathlib.Path(__file__).parents[1].resolve())
                    super().__init__(config, node_path=__name__, pkd_base_dir=pkd_base_dir)

                def run(self):
                    return {}
            """
        )
        fp.write(content)


def create_node_config(config_dir, node_name):
    config_text = {"root": None, "input": ["source"], "output": ["end"]}
    node_config_file = f"{node_name}.yml"

    with open(config_dir / node_config_file, "w") as fp:
        yaml.dump(config_text, fp)


def prepare_environment(paths):
    sys.path.append(str(paths.module_dir))

    paths.pkd_node_dir.mkdir(parents=True)
    paths.custom_node_dir.mkdir(parents=True)
    paths.pkd_node_config_dir.mkdir(parents=True)
    paths.custom_node_config_dir.mkdir(parents=True)

    create_pipeline_yaml(paths.pipeline_path, NODES)

    create_node_python(paths.pkd_node_dir, PKD_NODE_NAME)
    create_node_python(paths.custom_node_dir, CUSTOM_NODE_NAME)

    create_node_config(paths.pkd_node_config_dir, PKD_NODE_NAME)
    create_node_config(paths.custom_node_config_dir, CUSTOM_NODE_NAME)


def replace_init_node(path_to_node, node_name, config_loader, config_updates):
    return [path_to_node, node_name, config_loader, config_updates]


def replace_instantiate_nodes_return_none():
    return None


def replace_instantiate_nodes():
    instantiated_nodes = []

    node_path = PKD_NODE

    node = importlib.import_module(node_path)
    config = {"root": None, "input": ["source"], "output": ["end"]}

    instantiated_nodes.append(node.Node(config))

    return instantiated_nodes


@pytest.mark.usefixtures("tmp_dir")
class TestDeclarativeLoader:
    def test_loaded_node_list(self, declarative_loader):
        loaded_nodes = declarative_loader.node_list

        for idx, (node, config_updates) in enumerate(loaded_nodes):
            if isinstance(NODES["nodes"][idx], dict):
                assert {node: config_updates} == NODES["nodes"][idx]
            else:
                assert node == NODES["nodes"][idx]

    def test_get_custom_name_from_node_list(self, declarative_loader):
        custom_folder_name = declarative_loader._get_custom_name_from_node_list()

        assert custom_folder_name == CUSTOM_NODE_NAME

    def test_instantiate_nodes(self, declarative_loader):
        pkd_node_default = [
            "peekingduck.nodes.",
            PKD_NODE,
            declarative_loader.config_loader,
            None,
        ]
        pkd_node_edit = [
            "peekingduck.nodes.",
            PKD_NODE,
            declarative_loader.config_loader,
            [{"setting": True}],
        ]
        custom_node = [
            f"{CUSTOM_NODE_NAME}.",
            CUSTOM_NODE,
            declarative_loader.custom_config_loader,
            None,
        ]
        ground_truth = [pkd_node_default, pkd_node_edit, custom_node]

        with mock.patch(
            "peekingduck.declarative_loader.DeclarativeLoader._init_node",
            wraps=replace_init_node,
        ):
            instantiated_nodes = declarative_loader._instantiate_nodes()

            for node_num, node in enumerate(instantiated_nodes):
                for idx, output in enumerate(node):
                    assert output == ground_truth[node_num][idx]

    def test_init_node_pkd(self, declarative_loader):
        path_to_node = ""
        node_name = PKD_NODE
        config_loader = declarative_loader.config_loader
        config_updates = None

        init_node = declarative_loader._init_node(
            path_to_node, node_name, config_loader, config_updates
        )

        assert init_node._name == node_name
        assert init_node.inputs == ["source"]
        assert init_node.outputs == ["end"]

    def test_init_node_custom(self, declarative_loader):
        path_to_node = f"{CUSTOM_NODES_DIR_NAME}."
        node_name = CUSTOM_NODE
        config_loader = declarative_loader.custom_config_loader
        config_updates = None

        init_node = declarative_loader._init_node(
            path_to_node, node_name, config_loader, config_updates
        )

        assert init_node._name == f"{path_to_node}{node_name}"
        assert init_node.inputs == ["source"]
        assert init_node.outputs == ["end"]

    def test_init_node_edit(self, declarative_loader):
        path_to_node = ""
        node_name = PKD_NODE
        config_loader = declarative_loader.config_loader
        config_updates = {"input": ["img"]}

        init_node = declarative_loader._init_node(
            path_to_node, node_name, config_loader, config_updates
        )

        assert init_node._name == node_name
        assert init_node.inputs == ["img"]
        assert init_node.outputs == ["end"]

    def test_edit_config(self, declarative_loader):
        node_name = "input.visual"
        orig_config = {
            "mirror_image": True,
            "resize": {"do_resizing": False, "width": 1280, "height": 720},
        }
        config_update = {
            "mirror_image": False,
            "resize": {"do_resizing": True},
            "invalid_key": 123,
        }
        ground_truth = {
            "mirror_image": False,
            "resize": {"do_resizing": True, "width": 1280, "height": 720},
        }

        orig_config = declarative_loader._edit_config(
            orig_config, config_update, node_name
        )

        assert orig_config["mirror_image"] == ground_truth["mirror_image"]
        assert (
            orig_config["resize"]["do_resizing"]
            == ground_truth["resize"]["do_resizing"]
        )
        # Ensure that config_update does not replace the "resize" sub-dict completely and
        # erase "width" or "height"
        assert "width" in orig_config["resize"]
        assert "invalid_key" not in ground_truth

    def test_obj_detection_label_to_id_all_int(self, declarative_loader):
        node_name = "model.yolo"
        orig_config = {
            "detect": [0],
        }
        config_update = {"detect": [0, 1, 2, 3, 5]}
        ground_truth = {"detect": [0, 1, 2, 3, 5]}

        test_config = declarative_loader._edit_config(
            orig_config, config_update, node_name
        )
        assert test_config["detect"] == ground_truth["detect"]

    def test_obj_detection_label_to_id_all_text(self, declarative_loader):
        node_name = "model.yolo"
        orig_config = {
            "detect": [0],
        }
        config_update = {"detect": ["person", "car", "bus", "cell phone", "oven"]}
        ground_truth = {"detect": [0, 2, 5, 67, 69]}

        test_config = declarative_loader._edit_config(
            orig_config, config_update, node_name
        )
        assert test_config["detect"] == ground_truth["detect"]

    def test_obj_detection_label_to_id_mix_int_and_text(self, declarative_loader):
        node_name = "model.yolo"
        orig_config = {
            "detect": [0],
        }
        config_update = {"detect": [4, "bicycle", 10, "laptop", "teddy bear"]}
        ground_truth = {"detect": [1, 4, 10, 63, 77]}

        test_config = declarative_loader._edit_config(
            orig_config, config_update, node_name
        )
        assert test_config["detect"] == ground_truth["detect"]

    def test_obj_detection_label_to_id_mix_int_and_text_duplicates(
        self, declarative_loader
    ):
        node_name = "model.yolo"
        orig_config = {
            "detect": [0],
        }
        config_update = {
            "detect": [
                4,
                "bicycle",
                10,
                "laptop",
                "teddy bear",
                "aeroplane",
                63,
                10,
            ]
        }
        ground_truth = {"detect": [1, 4, 10, 63, 77]}

        test_config = declarative_loader._edit_config(
            orig_config, config_update, node_name
        )
        assert test_config["detect"] == ground_truth["detect"]

    def test_obj_detection_label_to_id_mix_int_and_text_errors(
        self, declarative_loader
    ):
        node_name = "model.yolo"
        orig_config = {
            "detect": [0],
        }
        config_update = {
            "detect": [
                4,
                "bicycle",
                10,
                "laptop",
                "teddy bear",
                "aeroplane",
                63,
                10,
                "pokemon",
                "scary monster",
            ]
        }
        ground_truth = {"detect": [0, 1, 4, 10, 63, 77]}

        test_config = declarative_loader._edit_config(
            orig_config, config_update, node_name
        )
        assert test_config["detect"] == ground_truth["detect"]

    def test_get_pipeline(self, declarative_loader):
        with mock.patch(
            "peekingduck.declarative_loader.DeclarativeLoader._instantiate_nodes",
            wraps=replace_instantiate_nodes,
        ):
            pipeline = declarative_loader.get_pipeline()
            assert pipeline.nodes[0]._name == PKD_NODE
            assert pipeline.nodes[0].inputs == ["source"]
            assert pipeline.nodes[0].outputs == ["end"]

    def test_get_pipeline_error(self, declarative_loader):
        with mock.patch(
            "peekingduck.declarative_loader.DeclarativeLoader._instantiate_nodes",
            wraps=replace_instantiate_nodes_return_none,
        ), pytest.raises(TypeError):
            declarative_loader.get_pipeline()
