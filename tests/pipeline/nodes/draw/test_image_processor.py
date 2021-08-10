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
Test for image processor node
"""
import pytest
import numpy as np
from peekingduck.pipeline.nodes.draw.image_processor import Node

@pytest.fixture
def draw_image_processor():
    node = Node({"input": ["img"],
                 "output": ["img"],
                 "brightness": 0,
                 "contrast": 1
                 })
    return node

@pytest.fixture
def draw_image_processor_brighter():
    node = Node({"input": ["img"],
                 "output": ["img"],
                 "brightness": 20,
                 "contrast": 1
                 })
    return node

@pytest.fixture
def draw_image_processor_contrast():
    node = Node({"input": ["img"],
                 "output": ["img"],
                 "brightness": 0,
                 "contrast": 2
                 })
    return node


class TestImageProcessor:
    def test_no_change(self, draw_image_processor, create_image):
        original_img = create_image((28, 28, 3))
        input1 = {
        "img": original_img
        }
        results = draw_image_processor.run(input1)
        np.testing.assert_equal(original_img, results['img'])

    # formula: processed image = contrast * image + brightness
    def test_brighten_image(self, draw_image_processor_brighter):
        original_img = np.ones((28, 28, 3))
        input1 = {
        "img": original_img
        }
        results = draw_image_processor_brighter.run(input1)

        assert original_img.shape == results['img'].shape
        np.testing.assert_raises(AssertionError, np.testing.assert_equal,
                                 original_img, results['img'])
        np.testing.assert_equal(results['img'][0][0], original_img[0][0] + 20)

    def test_increase_contrast(self, draw_image_processor_contrast):
        original_img = np.ones((28, 28, 3))
        input1 = {
        "img": original_img
        }
        results = draw_image_processor_contrast.run(input1)

        assert original_img.shape == results['img'].shape
        np.testing.assert_raises(AssertionError, np.testing.assert_equal,
                                 original_img, results['img'])
        np.testing.assert_equal(results['img'][0][0] / 2, original_img[0][0])
