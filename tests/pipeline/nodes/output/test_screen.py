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

import cv2
import pytest

from peekingduck.pipeline.nodes.output.screen import Node


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

    def test_viewer_disables_screen(self):
        node = Node(
            {
                "window_name": "PeekingDuck",
                "window_size": {"do_resizing": False, "width": 1280, "height": 720},
                "window_loc": {"x": 0, "y": 0},
                "pkd_viewer": True,
            }
        )
        with pytest.raises(cv2.error) as excinfo:
            cv2.getWindowProperty(node.window_name, cv2.WND_PROP_VISIBLE)
        assert "please create a window" in str(excinfo)
