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
Test for draw zones node
"""

import numpy as np
import pytest

from peekingduck.pipeline.nodes.draw.zones import Node


@pytest.fixture
def draw_zones():
    node = Node({"input": ["zones", "img"], "output": ["none"]})
    return node


class TestZones:
    def test_no_zones(self, draw_zones, create_image):
        original_img = create_image((28, 28, 3))
        output_img = original_img.copy()
        input1 = {
            "img": output_img,
            "zones": [],
        }
        draw_zones.run(input1)
        assert original_img.shape == output_img.shape
        np.testing.assert_equal(original_img, output_img)

    # formula: processed image = contrast * image + brightness
    def test_zones(self, draw_zones, create_image):
        original_img = create_image((28, 28, 3))
        output_img = original_img.copy()
        input1 = {"img": output_img, "zones": [[(2, 2), (10, 10), (10, 20), (2, 20)]]}
        draw_zones.run(input1)

        assert original_img.shape == output_img.shape
        np.testing.assert_raises(
            AssertionError, np.testing.assert_equal, original_img, output_img
        )
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_equal,
            original_img[2][2],
            output_img[2][2],
        )
