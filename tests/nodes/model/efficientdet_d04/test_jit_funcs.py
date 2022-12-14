# Copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import numpy.testing as npt
import pytest

from peekingduck.nodes.model.efficientdet_d04.efficientdet_files import (
    jit_funcs,
    nojit_funcs,
)
from peekingduck.nodes.model.efficientdet_d04.efficientdet_files.constants import (
    IMG_MEAN,
    IMG_STD,
)


@pytest.fixture(name="image_and_padding")
def fixture_image_and_padding():
    image = np.random.randint(10, size=(5, 4, 3))
    expected_image = np.zeros((1, 10, 11, 3), dtype=np.float32)
    expected_image[0, : image.shape[0], : image.shape[1]] = (
        image.astype(np.float32) / 255.0 - IMG_MEAN
    ) / IMG_STD
    pad_height = expected_image.shape[1] - image.shape[0]
    pad_width = expected_image.shape[2] - image.shape[1]

    return image, pad_height, pad_width, expected_image


def test_normalize_and_pad_with_jit(image_and_padding):
    image, pad_height, pad_width, expected_image = image_and_padding

    padded_image = jit_funcs.normalize_and_pad(image, pad_height, pad_width)

    npt.assert_allclose(padded_image, expected_image, rtol=1e-6)


def test_normalize_and_pad_without_jit(image_and_padding):
    image, pad_height, pad_width, expected_image = image_and_padding

    padded_image = nojit_funcs.normalize_and_pad(image, pad_height, pad_width)

    npt.assert_allclose(padded_image, expected_image, rtol=1e-6)
