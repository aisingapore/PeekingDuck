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

"""
Test for draw heat_map node
"""

import cv2
import numpy as np
import pytest

from peekingduck.nodes.draw.heat_map import Node
from tests.conftest import TEST_IMAGES_DIR

TEST_IMAGE = str(TEST_IMAGES_DIR / "crowd1.jpg")


@pytest.fixture
def draw_heat_map_node():
    node = Node({"input": ["density_map", "img"], "output": ["img"]})
    return node


class TestHeatmap:
    def test_no_heat_map(self, draw_heat_map_node):
        original_img = cv2.imread(TEST_IMAGE)
        output_img = original_img.copy()

        input = {"img": output_img, "density_map": np.zeros((768, 1024, 3))}

        output_img = draw_heat_map_node.run(input)
        np.testing.assert_equal(original_img, output_img["img"])

    def test_heat_map(self, draw_heat_map_node):
        original_img = cv2.imread(TEST_IMAGE)
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
