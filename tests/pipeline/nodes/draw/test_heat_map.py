# Copyright 2021 AI Singapore
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

"""
Test for draw heat_map node
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from peekingduck.pipeline.nodes.draw.heat_map import Node

TEST_IMAGE = ["crowd1.jpg"]
# path to reach 4 file levels up from test_heat_map.py
PKD_DIR = Path(__file__).resolve().parents[3]


@pytest.fixture(params=TEST_IMAGE)
def test_image(request):
    test_img_dir = PKD_DIR.parent / "images" / "testing"

    yield test_img_dir / request.param


@pytest.fixture
def draw_heat_map_node():
    node = Node({"input": ["density_map", "img"], "output": ["img"]})
    return node


class TestHeatmap:
    def test_no_heat_map(self, draw_heat_map_node, test_image):
        original_img = cv2.imread(str(test_image))
        output_img = original_img.copy()

        input = {"img": output_img, "density_map": np.zeros((768, 1024, 3))}

        output_img = draw_heat_map_node.run(input)
        np.testing.assert_equal(original_img, output_img["img"])

    def test_heat_map(self, draw_heat_map_node, test_image):
        original_img = cv2.imread(str(test_image))
        output_img = original_img.copy()

        input = {"img": output_img, "density_map": np.random.rand(768, 1024, 3)}

        output_img = draw_heat_map_node.run(input)

        # does not fail if the images are different
        # after applying heat_map
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            original_img,
            output_img["img"],
        )
