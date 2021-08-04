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
Test for draw bottom midpoint node
"""
import pytest
import numpy as np
from peekingduck.pipeline.nodes.draw.btm_midpoint import Node

@pytest.fixture
def draw_btm_mipoint():
    node = Node({"input": ["btm_midpoint", "img"],
                 "output": ["none"]
                 })
    return node


class TestBtmMidpoint:
    def test_no_btm_midpoint(self, draw_btm_mipoint, create_image):
        no_pts = []
        original_img = create_image((28, 28, 3))
        output_img = original_img.copy()
        input1 = {
        "btm_midpoint": no_pts,
        "img": output_img
        }
        draw_btm_mipoint.run(input1)
        np.testing.assert_equal(original_img, output_img)

    def test_btm_midpoint(self, draw_btm_mipoint, create_image):
        btm_midpoitns = [(0,0), (25, 25)]
        original_img = create_image((28, 28, 3))
        output_img = original_img.copy()
        input1 = {
        "btm_midpoint": btm_midpoitns,
        "img": output_img
        }
        draw_btm_mipoint.run(input1)

        assert original_img.shape == output_img.shape
        np.testing.assert_raises(AssertionError, np.testing.assert_equal,
                                 original_img, output_img)
        np.testing.assert_equal(output_img[0][0], np.array([156, 223, 244]))
        np.testing.assert_equal(output_img[25][25], np.array([156, 223, 244]))
