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
Test for draw tag node
"""
import pytest
import numpy as np
from peekingduck.pipeline.nodes.draw.tag import Node

@pytest.fixture
def draw_tag():
    node = Node({'input': ["bboxes", "obj_tags", "img"],
                 'output': ["none"]
                })
    return node


class TestTag:
    def test_no_tags(self, draw_tag, create_image):
        original_img = create_image((28, 28, 3))
        output_img = original_img.copy()
        input1 = {
        "img": output_img,
        "bboxes": [],
        "obj_tags": []
        }
        draw_tag.run(input1)
        assert original_img.shape == output_img.shape
        np.testing.assert_equal(original_img, output_img)


    # formula: processed image = contrast * image + brightness
    def test_tag(self, draw_tag, create_image):
        original_img = create_image((400, 400, 3))
        output_img = original_img.copy()
        input1 = {
        "img": output_img,
        "bboxes": [np.array([0, 0.5, 1, 1])],
        "obj_tags": ["TOO CLOSE!"]
        }
        draw_tag.run(input1)

        assert original_img.shape == output_img.shape
        np.testing.assert_raises(AssertionError, np.testing.assert_equal,
                                 original_img, output_img)
