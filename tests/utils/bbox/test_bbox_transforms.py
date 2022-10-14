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

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
import torch

from peekingduck.utils.bbox.transforms import (
    tlwh2xyah,
    tlwh2xyxy,
    tlwh2xyxyn,
    xywh2xyxy,
    xywhn2xyxy,
    xywhn2xyxyn,
    xyxy2tlwh,
    xyxy2xywh,
    xyxy2xywhn,
    xyxy2xyxyn,
    xyxyn2tlwh,
    xyxyn2xywhn,
    xyxyn2xyxy,
)
from tests.conftest import get_groundtruth
from tests.utils.bbox.utils import expand_dim

GT_RESULTS = get_groundtruth(Path(__file__).resolve())

# the variables below corresponds to parametrize's values
TRANSFORMS_REQUIRE_HEIGHT_WIDTH = [
    "xyxy2xywhn",
    "xywhn2xyxy",
    "xyxy2xyxyn",
    "xyxyn2xyxy",
    "xyxyn2tlwh",
    "tlwh2xyxyn",
]
TRANSFORMS_DO_NOT_REQUIRE_HEIGHT_WIDTH = [
    "tlwh2xyah",
    "xywhn2xyxyn",
    "xyxyn2xywhn",
    "xyxy2xywh",
    "xywh2xyxy",
    "tlwh2xyxy",
    "xyxy2tlwh",
]
CONVERT_TYPES = [np.array, torch.tensor]
NUM_DIMS = [0, 1, 2]

# tolerance to assert_allclose
ATOL, RTOL = 1e-4, 1e-07


@pytest.fixture(scope="module")
def gt_bboxes():
    tlwh, xyah, xywh, xywhn, xyxy, xyxyn = (
        GT_RESULTS["tlwh"],
        GT_RESULTS["xyah"],
        GT_RESULTS["xywh"],
        GT_RESULTS["xywhn"],
        GT_RESULTS["xyxy"],
        GT_RESULTS["xyxyn"],
    )
    return {
        "tlwh2xyah": [tlwh, xyah, tlwh2xyah],
        "tlwh2xyxyn": [tlwh, xyxyn, tlwh2xyxyn],
        "tlwh2xyxy": [tlwh, xyxy, tlwh2xyxy],
        "xywh2xyxy": [xywh, xyxy, xywh2xyxy],
        "xywhn2xyxyn": [xywhn, xyxyn, xywhn2xyxyn],
        "xywhn2xyxy": [xywhn, xyxy, xywhn2xyxy],
        "xyxy2xywhn": [xyxy, xywhn, xyxy2xywhn],
        "xyxy2tlwh": [xyxy, tlwh, xyxy2tlwh],
        "xyxy2xywh": [xyxy, xywh, xyxy2xywh],
        "xyxy2xyxyn": [xyxy, xyxyn, xyxy2xyxyn],
        "xyxyn2xyxy": [xyxyn, xyxy, xyxyn2xyxy],
        "xyxyn2tlwh": [xyxyn, tlwh, xyxyn2tlwh],
        "xyxyn2xywhn": [xyxyn, xywhn, xyxyn2xywhn],
    }


@pytest.mark.parametrize("convert_type", CONVERT_TYPES)
class TestBboxTransforms:
    @pytest.mark.parametrize("conversion_name", TRANSFORMS_REQUIRE_HEIGHT_WIDTH)
    def test_correct_return_type_require_height_width(
        self, gt_bboxes, convert_type, conversion_name
    ):
        from_bbox, _, conversion_fn = gt_bboxes[conversion_name]
        from_bbox = convert_type(from_bbox)

        to_bbox = conversion_fn(
            from_bbox, height=GT_RESULTS["height"], width=GT_RESULTS["width"]
        )

        assert isinstance(to_bbox, type(from_bbox))

    @pytest.mark.parametrize("conversion_name", TRANSFORMS_DO_NOT_REQUIRE_HEIGHT_WIDTH)
    def test_correct_return_type_do_not_require_height_width(
        self, gt_bboxes, convert_type, conversion_name
    ):
        from_bbox, _, conversion_fn = gt_bboxes[conversion_name]
        from_bbox = convert_type(from_bbox)

        to_bbox = conversion_fn(from_bbox)

        assert isinstance(to_bbox, type(from_bbox))

    @pytest.mark.parametrize("num_dims", NUM_DIMS)
    @pytest.mark.parametrize("conversion_name", TRANSFORMS_REQUIRE_HEIGHT_WIDTH)
    def test_consistent_in_out_dims_require_height_width(
        self, gt_bboxes, convert_type, num_dims, conversion_name
    ):
        from_bbox, _, conversion_fn = gt_bboxes[conversion_name]
        from_bbox = convert_type(from_bbox)
        from_bbox = expand_dim(from_bbox, num_dims)

        to_bbox = conversion_fn(
            from_bbox, height=GT_RESULTS["height"], width=GT_RESULTS["width"]
        )

        assert from_bbox.shape == to_bbox.shape

    @pytest.mark.parametrize("num_dims", NUM_DIMS)
    @pytest.mark.parametrize("conversion_name", TRANSFORMS_DO_NOT_REQUIRE_HEIGHT_WIDTH)
    def test_consistent_in_out_dims_do_not_require_height_width(
        self, gt_bboxes, convert_type, num_dims, conversion_name
    ):
        from_bbox, _, conversion_fn = gt_bboxes[conversion_name]
        from_bbox = convert_type(from_bbox)
        from_bbox = expand_dim(from_bbox, num_dims)

        to_bbox = conversion_fn(from_bbox)

        assert from_bbox.shape == to_bbox.shape

    @pytest.mark.parametrize("num_dims", NUM_DIMS)
    @pytest.mark.parametrize("conversion_name", TRANSFORMS_REQUIRE_HEIGHT_WIDTH)
    def test_correct_transformation_require_height_width(
        self, gt_bboxes, convert_type, num_dims, conversion_name
    ):
        from_bbox, expected_bbox, conversion_fn = gt_bboxes[conversion_name]
        from_bbox = convert_type(from_bbox)
        from_bbox = expand_dim(from_bbox, num_dims)

        to_bbox = conversion_fn(
            from_bbox, height=GT_RESULTS["height"], width=GT_RESULTS["width"]
        )

        expected_bbox = expand_dim(convert_type(expected_bbox), num_dims)

        if isinstance(to_bbox, torch.Tensor):
            torch.testing.assert_allclose(to_bbox, expected_bbox, atol=ATOL, rtol=RTOL)
        else:
            npt.assert_allclose(to_bbox, expected_bbox, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize("num_dims", NUM_DIMS)
    @pytest.mark.parametrize("conversion_name", TRANSFORMS_DO_NOT_REQUIRE_HEIGHT_WIDTH)
    def test_correct_transformation_do_not_require_height_width(
        self, gt_bboxes, convert_type, num_dims, conversion_name
    ):
        from_bbox, expected_bbox, conversion_fn = gt_bboxes[conversion_name]
        from_bbox = convert_type(from_bbox)
        from_bbox = expand_dim(from_bbox, num_dims)

        to_bbox = conversion_fn(from_bbox)

        expected_bbox = expand_dim(convert_type(expected_bbox), num_dims)

        if isinstance(to_bbox, torch.Tensor):
            torch.testing.assert_allclose(to_bbox, expected_bbox, atol=ATOL, rtol=RTOL)
        else:
            npt.assert_allclose(to_bbox, expected_bbox, atol=ATOL, rtol=RTOL)
