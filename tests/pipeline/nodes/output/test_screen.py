# Copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import platform
from unittest import mock

import cv2
import pytest
import yaml

from peekingduck.pipeline.nodes.output.screen import Node
from tests.conftest import PKD_DIR


@pytest.fixture
def screen_config():
    with open(PKD_DIR / "configs" / "output" / "screen.yml") as infile:
        node_config = yaml.safe_load(infile)

    return node_config


@pytest.fixture
def opencv_instance():
    yield
    cv2.destroyAllWindows()


@pytest.mark.skipif(platform.system() != "Linux", reason="xvfb requires Linux")
@pytest.mark.usefixtures("opencv_instance")
class TestScreen:
    def test_move_on_first_run(self, screen_config, create_input_image):
        x = 100
        y = 200
        screen_config["window_loc"] = {"x": x, "y": y}

        node = Node(screen_config)
        filename = "image1.png"
        image = create_input_image(filename, (900, 800, 3))
        inputs = {"filename": filename, "img": image}

        with mock.patch("cv2.moveWindow") as mock_move_1:
            node.run(inputs)
        assert not node.first_run

        with mock.patch("cv2.moveWindow") as mock_move_2:
            node.run(inputs)

        assert mock_move_1.call_args == ((node.window_name, x, y),)
        assert mock_move_2.call_args is None

    def test_resize_to_config(self, screen_config, create_input_image):
        width = 800
        height = 900
        screen_config["window_size"] = {
            "do_resizing": True,
            "height": height,
            "width": width,
        }
        node = Node(screen_config)
        filename = "image1.png"
        image = create_input_image(filename, (height // 2, width // 2, 3))
        inputs = {"filename": filename, "img": image}
        with mock.patch("cv2.resize", wraps=cv2.resize) as mock_resize:
            node.run(inputs)

        assert mock_resize.call_args[0][1] == (height, width)

    def test_no_resize_by_default(self, screen_config, create_input_image):
        width = 800
        height = 900

        node = Node(screen_config)
        filename = "image1.png"
        image = create_input_image(filename, (height // 2, width // 2, 3))
        inputs = {"filename": filename, "img": image}
        with mock.patch("cv2.resize", wraps=cv2.resize) as mock_resize:
            node.run(inputs)

        assert mock_resize.call_args is None
