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

from peekingduck.pipeline.nodes.output.screen import MIN_DISPLAY_SIZE, Node
from tests.conftest import PKD_DIR


@pytest.fixture
def screen_config():
    with open(PKD_DIR / "configs" / "output" / "screen.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["pkd_viewer"] = False

    return node_config


@pytest.fixture
def opencv_instance():
    yield
    cv2.destroyAllWindows()


@pytest.mark.skipif(platform.system() != "Linux", reason="xvfb requires Linux")
@pytest.mark.usefixtures("opencv_instance")
class TestScreen:
    def test_default_screen(self):
        node = Node()
        prop = cv2.getWindowProperty(node.window_name, cv2.WND_PROP_VISIBLE)
        assert int(prop) == 1

    def test_viewer_disables_screen(self, screen_config):
        screen_config["pkd_viewer"] = True
        node = Node(screen_config)
        with pytest.raises(cv2.error) as excinfo:
            cv2.getWindowProperty(node.window_name, cv2.WND_PROP_VISIBLE)
        assert "please create a window" in str(excinfo)

    def test_resize_to_image(self, create_input_image):
        node = Node()
        filename = "image1.png"
        width = 800
        height = 900
        image = create_input_image(filename, (height, width, 3))
        inputs = {"filename": filename, "img": image}
        with mock.patch("cv2.resizeWindow") as mock_resize:
            node.run(inputs)

        assert mock_resize.call_args == ((node.window_name, width, height),)

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
        with mock.patch("cv2.resizeWindow") as mock_resize:
            node.run(inputs)

        assert mock_resize.call_args == ((node.window_name, width, height),)

    def test_resize_to_min_size(self, screen_config, create_input_image):
        width = MIN_DISPLAY_SIZE // 2
        height = MIN_DISPLAY_SIZE // 2
        screen_config["window_size"] = {
            "do_resizing": True,
            "height": height,
            "width": width,
        }
        node = Node(screen_config)
        filename = "image1.png"
        image = create_input_image(filename, (height // 2, width // 2, 3))
        inputs = {"filename": filename, "img": image}
        with mock.patch("cv2.resizeWindow") as mock_resize:
            node.run(inputs)

        assert mock_resize.call_args == (
            (node.window_name, MIN_DISPLAY_SIZE, MIN_DISPLAY_SIZE),
        )
