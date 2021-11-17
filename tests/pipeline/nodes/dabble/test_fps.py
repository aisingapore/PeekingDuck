"""
Copyright 2021 AI Singapore

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest

from peekingduck.pipeline.nodes.dabble.fps import Node


@pytest.fixture
def fps_node():
    node = Node(
        {
            "input": ["pipeline_end"],
            "output": ["fps"],
            "fps_log_display": True,
            "fps_log_freq": 30,
            "dampen_fps": True,
        }
    )
    return node


class TestFPS:
    def test_type_fps(self, fps_node):
        input1 = {"pipeline_end": False}

        assert isinstance(fps_node.run(input1)["fps"], float)

    def test_positive_fps(self, fps_node):
        input1 = {"pipeline_end": False}

        assert fps_node.run(input1)["fps"] > 0
