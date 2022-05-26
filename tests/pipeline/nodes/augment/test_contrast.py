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
Test for augment contrast node
"""

import numpy as np
import pytest

from peekingduck.pipeline.nodes.augment.contrast import Node


@pytest.fixture
def contrast_same():
    node = Node({"input": ["img"], "output": ["img"], "alpha": 1.0})
    return node


@pytest.fixture
def contrast_increase():
    node = Node({"input": ["img"], "output": ["img"], "alpha": 2.0})
    return node


class TestContrast:
    def test_no_change(self, contrast_same, create_image):
        original_img = create_image((28, 28, 3))
        input1 = {"img": original_img}
        results = contrast_same.run(input1)
        np.testing.assert_equal(original_img, results["img"])

    def test_increase_contrast(self, contrast_increase):
        original_img = np.ones(shape=(28, 28, 3), dtype=np.uint8)
        input1 = {"img": original_img}
        results = contrast_increase.run(input1)

        assert original_img.shape == results["img"].shape
        with pytest.raises(AssertionError):
            np.testing.assert_equal(original_img, results["img"])
        np.testing.assert_equal(results["img"][0][0], original_img[0][0] * 2)

    def test_overflow(self, contrast_increase):
        # Test positive overflow - any values that sum up to higher than 255 will
        # be clipped at 255
        bright_img = np.ones(shape=(28, 28, 3), dtype=np.uint8) * 250
        bright_input = {"img": bright_img}
        results = contrast_increase.run(bright_input)
        np.testing.assert_equal(results["img"][0][0], np.array([255, 255, 255]))

    def test_beta_range(self):
        with pytest.raises(ValueError) as excinfo:
            Node({"input": ["img"], "output": ["img"], "alpha": -0.5})
        assert str(excinfo.value) == "alpha must be between [0.0, 3.0]"

        with pytest.raises(ValueError) as excinfo:
            Node({"input": ["img"], "output": ["img"], "alpha": 3.1})
        assert str(excinfo.value) == "alpha must be between [0.0, 3.0]"
