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
import shutil
import yaml

from peekingduck.loaders import DeclarativeLoader


CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))


def create_configloader(run_config_path, custom_folder_path):
    declarative_loader = DeclarativeLoader(run_config_path, custom_folder_path)

    return config_loader


def create_run_config_yaml(nodes):

    run_config_file = "run_config.yml"
    run_config_path = os.path.join(CURRENT_PATH, run_config_file)

    with open(run_config_path, 'w') as outfile:
        yaml.dump(nodes, outfile, default_flow_style=False)

    return run_config_path


nodes = {"nodes": [{'input.live': [{"mirror_image": False}]},
                   'model.yolo',
                   'draw.bbox',
                   'custom.draw.fps',
                   'output.screen']
         }

# nodes = {"nodes": ['input.live',
#                    'model.yolo',
#                    'draw.bbox',
#                    'custom.draw.fps',
#                    'output.screen']
#          }

# run_config_path = create_run_config_yaml(nodes)

for idx, node in enumerate(nodes["nodes"]):
    print(nodes["nodes"][idx])


class TestDeclarativeLoader():

    def test_load_node_list(self):
        nodes = {"nodes": [{'input.live': [{"mirror_image": False}]},
                           'model.yolo',
                           'draw.bbox',
                           'custom.draw.fps',
                           'output.screen']
                 }

        run_config_path = create_run_config_yaml(nodes)
        declarative_loader = DeclarativeLoader(run_config_path)
        loaded_nodes = declarative_loader._load_node_list(run_config_path)

        os.remove(run_config_path)

        for idx, node in enumerate(loaded_nodes):
            assert node == nodes["nodes"][idx]

    # class TestConfigLoader():

    #     def test_config_loader_returns_correct_config_filepath(self):

    #         node = 'type.node'
    #         config_loader = create_configloader()
    #         filepath = config_loader._get_config_path(node)

    #         ground_truth = os.path.join(CURRENT_PATH,
    #                                     "configs",
    #                                     node.replace(".", "/"))
    #         ground_truth = ground_truth + ".yml"

    #         assert filepath == ground_truth

    #     def test_config_loader_load_correct_yaml(self):
    #         node = "input.test"
    #         data = {"input": "img",
    #                 "output": "img"}

    #         config_folder_dir = create_config_yaml(node, data)

    #         config_loader = create_configloader()
    #         config = config_loader.get(node)

    #         shutil.rmtree(config_folder_dir)

    #         for key in data.keys():
    #             assert data[key] == config[key]
