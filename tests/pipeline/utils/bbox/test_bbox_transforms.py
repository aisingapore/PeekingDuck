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

import pytest

# pylint: disable=unused-import
from peekingduck.pipeline.utils.bbox.transforms import (
    albu2yolo,
    tlwh2xyah,
    tlwh2xyxy,
    tlwh2xyxyn,
    voc2yolo,
    xywh2xyxy,
    xyxy2tlwh,
    xyxy2xywh,
    xyxy2xyxyn,
    xyxyn2tlwh,
    xyxyn2xyxy,
    yolo2albu,
    yolo2voc,
)
from tests.conftest import get_groundtruth
from tests.pipeline.utils.bbox.utils import expand_dim, list2numpy, list2torch

GT_RESULTS = get_groundtruth(Path(__file__).resolve())

# the variables below corresponds to parametrize's values
transforms_require_height_width = [
    "voc2yolo",
    "yolo2voc",
    "xyxy2xyxyn",
    "xyxyn2xyxy",
    "xyxyn2tlwh",
    "tlwh2xyxyn",
]
transforms_do_not_require_height_width = [
    "tlwh2xyah",
    "yolo2albu",
    "albu2yolo",
    "xyxy2xywh",
    "xywh2xyxy",
    "tlwh2xyxy",
    "xyxy2tlwh",
]
convert_types = [list2numpy, list2torch]
num_dims = [0, 1, 2]


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
        "tlwh2xyah": [coco, xyah],
        "yolo2albu": [yolo, albu],
        "albu2yolo": [albu, yolo],
        "voc2yolo": [voc, yolo],
        "yolo2voc": [yolo, voc],
        "tlwh2xyxy": [coco, voc],
        "xyxy2tlwh": [voc, coco],
        "xyxy2xywh": [voc, unnormalized_yolo],
        "xywh2xyxy": [unnormalized_yolo, voc],
        "xyxy2xyxyn": [voc, albu],
        "xyxyn2xyxy": [albu, voc],
        "xyxyn2tlwh": [albu, coco],
        "tlwh2xyxyn": [coco, albu],
    }


@pytest.mark.parametrize("convert_type", convert_types)
@pytest.mark.parametrize("conversion_name", transforms_require_height_width)
def test_correct_return_type_require_height_width(
    gt_bboxes, convert_type, conversion_name
):
    conversion_fn = globals()[conversion_name]

    from_bbox, _ = gt_bboxes[conversion_name]
    from_bbox = convert_type(from_bbox)

    to_bbox = conversion_fn(
        from_bbox, height=GT_RESULTS["height"], width=GT_RESULTS["width"]
    )

    assert isinstance(to_bbox, type(from_bbox))


@pytest.mark.parametrize("convert_type", convert_types)
@pytest.mark.parametrize("conversion_name", transforms_do_not_require_height_width)
def test_correct_return_type_do_not_require_height_width(
    gt_bboxes, convert_type, conversion_name
):
    conversion_fn = globals()[conversion_name]

    from_bbox, _ = gt_bboxes[conversion_name]
    from_bbox = convert_type(from_bbox)

    to_bbox = conversion_fn(from_bbox)

    assert isinstance(to_bbox, type(from_bbox))


@pytest.mark.parametrize("convert_type", convert_types)
@pytest.mark.parametrize("num_dims", num_dims)
@pytest.mark.parametrize("conversion_name", transforms_require_height_width)
def test_consistent_in_out_dims_require_height_width(
    gt_bboxes, convert_type, num_dims, conversion_name
):
    conversion_fn = globals()[conversion_name]

    from_bbox, _ = gt_bboxes[conversion_name]
    from_bbox = convert_type(from_bbox)
    from_bbox = expand_dim(from_bbox, num_dims)

    to_bbox = conversion_fn(
        from_bbox, height=GT_RESULTS["height"], width=GT_RESULTS["width"]
    )

    assert from_bbox.shape == to_bbox.shape


@pytest.mark.parametrize("convert_type", convert_types)
@pytest.mark.parametrize("num_dims", num_dims)
@pytest.mark.parametrize("conversion_name", transforms_do_not_require_height_width)
def test_consistent_in_out_dims_do_not_require_height_width(
    gt_bboxes, convert_type, num_dims, conversion_name
):
    conversion_fn = globals()[conversion_name]

    from_bbox, _ = gt_bboxes[conversion_name]
    from_bbox = convert_type(from_bbox)
    from_bbox = expand_dim(from_bbox, num_dims)

    to_bbox = conversion_fn(from_bbox)

    assert from_bbox.shape == to_bbox.shape


@pytest.mark.parametrize("convert_type", convert_types)
@pytest.mark.parametrize("num_dims", num_dims)
@pytest.mark.parametrize("conversion_name", transforms_require_height_width)
def test_correct_transformation_require_height_width(
    gt_bboxes, convert_type, num_dims, conversion_name
):
    conversion_fn = globals()[conversion_name]

    from_bbox, expected_bbox = gt_bboxes[conversion_name]
    from_bbox = convert_type(from_bbox)
    from_bbox = expand_dim(from_bbox, num_dims)

    to_bbox = conversion_fn(
        from_bbox, height=GT_RESULTS["height"], width=GT_RESULTS["width"]
    )

    expected_bbox = expand_dim(convert_type(expected_bbox), num_dims)

    assert expected_bbox.all() == pytest.approx(to_bbox.all(), abs=1e-4)


@pytest.mark.parametrize("convert_type", convert_types)
@pytest.mark.parametrize("num_dims", num_dims)
@pytest.mark.parametrize("conversion_name", transforms_do_not_require_height_width)
def test_correct_transformation_do_not_require_height_width(
    gt_bboxes, convert_type, num_dims, conversion_name
):
    conversion_fn = globals()[conversion_name]

    from_bbox, expected_bbox = gt_bboxes[conversion_name]
    from_bbox = convert_type(from_bbox)
    from_bbox = expand_dim(from_bbox, num_dims)

    to_bbox = conversion_fn(from_bbox)

    expected_bbox = expand_dim(convert_type(expected_bbox), num_dims)

    assert expected_bbox.all() == pytest.approx(to_bbox.all(), abs=1e-4)
