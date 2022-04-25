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

import numpy as np
import numpy.testing as npt
import pytest

from peekingduck.pipeline.nodes.model.hrnetv1.hrnet_files.preprocessing import (
    tlwh2xywh,
    crop_and_resize,
)


@pytest.fixture
def projected_bbox_arr():
    return np.array(
        [
            [359.5, 239.5, 143.8, 95.8],
            [0.0, 239.5, 719, 95.8],
            [359.5, 0.0, 143.8, 479.0],
        ]
    )


class TestPreprocessing:
    def test_tlwh2xywh(self, projected_bbox_arr):
        test_arr = projected_bbox_arr
        test_aspect_ratio = 1280 / 720
        actual_output = tlwh2xywh(test_arr, test_aspect_ratio)

        expected_output = np.array(
            [
                [431.4, 287.4, 170.31111111, 95.8],
                [359.5, 287.4, 719.0, 404.4375],
                [431.4, 239.5, 851.55555556, 479.0],
            ]
        )

        npt.assert_almost_equal(actual_output, expected_output)

    def test_crop_and_resize(self, create_image, projected_bbox_arr):
        test_img = create_image((720, 480))
        test_bboxes = projected_bbox_arr
        test_out_size = (256, 192)
        expected_output = np.array(
            [
                [[0.56171875, 0, 288.1], [0.0, 0.49895833, 192.1]],
                [[2.80859375, 0.0, -359.0], [0.0, 0.49895833, 192.1]],
                [[0.56171875, 0.0, 288.1], [0.0, 2.49479167, -239.0]],
            ]
        )
        _, actual_output = crop_and_resize(test_img, test_bboxes, test_out_size)

        npt.assert_almost_equal(actual_output, expected_output)
