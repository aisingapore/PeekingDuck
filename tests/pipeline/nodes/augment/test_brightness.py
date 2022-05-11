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
Test for augment brightness node
"""

import numpy as np
import pytest

from peekingduck.pipeline.nodes.augment.brightness import Node


@pytest.fixture
def brightness_same():
    node = Node({"input": ["img"], "output": ["img"], "beta": 0})
    return node


@pytest.fixture
def brightness_increase():
    node = Node({"input": ["img"], "output": ["img"], "beta": 50})
    return node


@pytest.fixture
def brightness_decrease():
    node = Node({"input": ["img"], "output": ["img"], "beta": -50})
    return node


class TestBrightness:
    def test_no_change(self, brightness_same, create_image):
        original_img = create_image((28, 28, 3))
        input1 = {"img": original_img}
        results = brightness_same.run(input1)
        np.testing.assert_equal(original_img, results["img"])

    def test_brighten_image(self, brightness_increase):
        original_img = np.ones(shape=(28, 28, 3), dtype=np.uint8)
        input1 = {"img": original_img.copy()}
        results = brightness_increase.run(input1)

        assert original_img.shape == results["img"].shape
        with pytest.raises(AssertionError):
            np.testing.assert_equal(original_img, results["img"])
        np.testing.assert_equal(results["img"][0][0], original_img[0][0] + 50)

    def test_darken_image(self, brightness_decrease):

        original_img = np.ones(shape=(28, 28, 3), dtype=np.uint8) * 100
        print(original_img[0][0])
        input1 = {"img": original_img.copy()}
        results = brightness_decrease.run(input1)
        print(results["img"][0][0], original_img[0][0])

        assert original_img.shape == results["img"].shape
        with pytest.raises(AssertionError):
            np.testing.assert_equal(original_img, results["img"])
        np.testing.assert_equal(results["img"][0][0], original_img[0][0] - 50)

    def test_overflow(self, brightness_increase, brightness_decrease):
        # Test positive overflow - any values that sum up to higher than 255 will
        # be clipped at 255
        bright_img = np.ones(shape=(28, 28, 3), dtype=np.uint8) * 250
        bright_input = {"img": bright_img}
        results = brightness_increase.run(bright_input)
        np.testing.assert_equal(results["img"][0][0], np.array([255, 255, 255]))

        # Test negative overflow - any values that when subtracted are negative, will
        # be clipped at 0 instead
        dark_img = np.ones(shape=(28, 28, 3), dtype=np.uint8)
        dark_input = {"img": dark_img}
        results = brightness_decrease.run(dark_input)
        np.testing.assert_equal(results["img"][0][0], np.array([0, 0, 0]))

    def test_beta_range(self):
        with pytest.raises(ValueError) as excinfo:
            Node({"input": ["img"], "output": ["img"], "beta": -101})
        assert str(excinfo.value) == "beta must be between [-100.0, 100.0]"

        with pytest.raises(ValueError) as excinfo:
            Node({"input": ["img"], "output": ["img"], "beta": 101})
        assert str(excinfo.value) == "beta must be between [-100.0, 100.0]"
