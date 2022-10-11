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
import torch.testing as tpt
from peekingduck.pipeline.utils.bbox.transforms import (
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
from tests.pipeline.utils.bbox.utils import expand_dim, list2numpy, list2torch

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
CONVERT_TYPES = [list2numpy, list2torch]
NUM_DIMS = [0, 1, 2]

# tolerance to assert_allclose
ATOL, RTOL = 1e-4, 1e-07


@pytest.fixture(scope="module")
def gt_bboxes():
    xyah, voc, albu, coco, yolo, unnormalized_yolo = (
        GT_RESULTS["xyah"],
        GT_RESULTS["voc"],
        GT_RESULTS["albu"],
        GT_RESULTS["coco"],
        GT_RESULTS["yolo"],
        GT_RESULTS["unnormalized_yolo"],
    )
    return {
        "tlwh2xyah": [coco, xyah, tlwh2xyah],
        "xywhn2xyxyn": [yolo, albu, xywhn2xyxyn],
        "xyxyn2xywhn": [albu, yolo, xyxyn2xywhn],
        "xyxy2xywhn": [voc, yolo, xyxy2xywhn],
        "xywhn2xyxy": [yolo, voc, xywhn2xyxy],
        "tlwh2xyxy": [coco, voc, tlwh2xyxy],
        "xyxy2tlwh": [voc, coco, xyxy2tlwh],
        "xyxy2xywh": [voc, unnormalized_yolo, xyxy2xywh],
        "xywh2xyxy": [unnormalized_yolo, voc, xywh2xyxy],
        "xyxy2xyxyn": [voc, albu, xyxy2xyxyn],
        "xyxyn2xyxy": [albu, voc, xyxyn2xyxy],
        "xyxyn2tlwh": [albu, coco, xyxyn2tlwh],
        "tlwh2xyxyn": [coco, albu, tlwh2xyxyn],
    }


@pytest.mark.parametrize("convert_type", CONVERT_TYPES)
@pytest.mark.parametrize("conversion_name", TRANSFORMS_REQUIRE_HEIGHT_WIDTH)
def test_correct_return_type_require_height_width(
    gt_bboxes, convert_type, conversion_name
):
    from_bbox, _, conversion_fn = gt_bboxes[conversion_name]
    from_bbox = convert_type(from_bbox)

    to_bbox = conversion_fn(
        from_bbox, height=GT_RESULTS["height"], width=GT_RESULTS["width"]
    )

    assert isinstance(to_bbox, type(from_bbox))


@pytest.mark.parametrize("convert_type", CONVERT_TYPES)
@pytest.mark.parametrize("conversion_name", TRANSFORMS_DO_NOT_REQUIRE_HEIGHT_WIDTH)
def test_correct_return_type_do_not_require_height_width(
    gt_bboxes, convert_type, conversion_name
):
    from_bbox, _, conversion_fn = gt_bboxes[conversion_name]
    from_bbox = convert_type(from_bbox)

    to_bbox = conversion_fn(from_bbox)

    assert isinstance(to_bbox, type(from_bbox))


@pytest.mark.parametrize("convert_type", CONVERT_TYPES)
@pytest.mark.parametrize("num_dims", NUM_DIMS)
@pytest.mark.parametrize("conversion_name", TRANSFORMS_REQUIRE_HEIGHT_WIDTH)
def test_consistent_in_out_dims_require_height_width(
    gt_bboxes, convert_type, num_dims, conversion_name
):
    from_bbox, _, conversion_fn = gt_bboxes[conversion_name]
    from_bbox = convert_type(from_bbox)
    from_bbox = expand_dim(from_bbox, num_dims)

    to_bbox = conversion_fn(
        from_bbox, height=GT_RESULTS["height"], width=GT_RESULTS["width"]
    )

    assert from_bbox.shape == to_bbox.shape


@pytest.mark.parametrize("convert_type", CONVERT_TYPES)
@pytest.mark.parametrize("num_dims", NUM_DIMS)
@pytest.mark.parametrize("conversion_name", TRANSFORMS_DO_NOT_REQUIRE_HEIGHT_WIDTH)
def test_consistent_in_out_dims_do_not_require_height_width(
    gt_bboxes, convert_type, num_dims, conversion_name
):
    from_bbox, _, conversion_fn = gt_bboxes[conversion_name]
    from_bbox = convert_type(from_bbox)
    from_bbox = expand_dim(from_bbox, num_dims)

    to_bbox = conversion_fn(from_bbox)

    assert from_bbox.shape == to_bbox.shape


@pytest.mark.parametrize("convert_type", CONVERT_TYPES)
@pytest.mark.parametrize("num_dims", NUM_DIMS)
@pytest.mark.parametrize("conversion_name", TRANSFORMS_REQUIRE_HEIGHT_WIDTH)
def test_correct_transformation_require_height_width(
    gt_bboxes, convert_type, num_dims, conversion_name
):
    from_bbox, expected_bbox, conversion_fn = gt_bboxes[conversion_name]
    from_bbox = convert_type(from_bbox)
    from_bbox = expand_dim(from_bbox, num_dims)

    to_bbox = conversion_fn(
        from_bbox, height=GT_RESULTS["height"], width=GT_RESULTS["width"]
    )

    expected_bbox = expand_dim(convert_type(expected_bbox), num_dims)

    npt.assert_allclose(to_bbox, expected_bbox, atol=ATOL, rtol=RTOL) if isinstance(
        to_bbox, np.ndarray
    ) else tpt.assert_allclose(to_bbox, expected_bbox, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("convert_type", CONVERT_TYPES)
@pytest.mark.parametrize("num_dims", NUM_DIMS)
@pytest.mark.parametrize("conversion_name", TRANSFORMS_DO_NOT_REQUIRE_HEIGHT_WIDTH)
def test_correct_transformation_do_not_require_height_width(
    gt_bboxes, convert_type, num_dims, conversion_name
):
    from_bbox, expected_bbox, conversion_fn = gt_bboxes[conversion_name]
    from_bbox = convert_type(from_bbox)
    from_bbox = expand_dim(from_bbox, num_dims)

    to_bbox = conversion_fn(from_bbox)

    expected_bbox = expand_dim(convert_type(expected_bbox), num_dims)

    npt.assert_allclose(to_bbox, expected_bbox, atol=ATOL, rtol=RTOL) if isinstance(
        to_bbox, np.ndarray
    ) else tpt.assert_allclose(to_bbox, expected_bbox, atol=ATOL, rtol=RTOL)
