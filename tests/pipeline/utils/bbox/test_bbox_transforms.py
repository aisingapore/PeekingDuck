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

from typing import Union

import numpy as np
import pytest
import torch

# pylint: disable=unused-import
from peekingduck.pipeline.utils.bbox.transforms import (
    albu2yolo,
    clone,
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

HEIGHT, WIDTH = 480, 640

xyah = [259, 403.5, 2.7521367, 117]
voc = [98, 345, 420, 462]
albu = [0.153125, 0.71875, 0.65625, 0.9625]
coco = [98, 345, 322, 117]
yolo = [0.4046875, 0.840625, 0.503125, 0.24375]
unnormalized_yolo = [259.0, 403.5, 322.0, 117.0]


def list2numpy(list_: list) -> np.ndarray:
    return np.asarray(list_)


def list2torch(list_: list) -> torch.Tensor:
    return torch.tensor(list_)


def expand_dim(
    bboxes: Union[np.ndarray, torch.Tensor],
    num_dims: int,
) -> Union[np.ndarray, torch.Tensor]:
    """Expand the dimension of bboxes (first in) by num_dims.

    Note:
        np.expand_dims will not work for tuple dim numpy < 1.18.0 which
        is not the version in our cicd.

    Args:
        bboxes (Union[np.ndarray, torch.Tensor]): The input bboxes.
        num_dims (int): The number of dimensions to expand.

    Returns:
        (Union[np.ndarray, torch.Tensor]): The bboxes with expanded dimensions.
    """
    bboxes = clone(bboxes)
    return bboxes[(None,) * num_dims]


@pytest.fixture(scope="module")
def gt_bboxes():
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


@pytest.mark.parametrize("convert_type", convert_types)
@pytest.mark.parametrize("conversion_name", transforms_require_height_width)
def test_correct_return_type_require_height_width(
    gt_bboxes, convert_type, conversion_name
):
    conversion_fn = globals()[conversion_name]

    from_bbox, _ = gt_bboxes[conversion_name]
    from_bbox = convert_type(from_bbox)

    to_bbox = conversion_fn(from_bbox, height=HEIGHT, width=WIDTH)

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

    to_bbox = conversion_fn(from_bbox, height=HEIGHT, width=WIDTH)

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

    to_bbox = conversion_fn(from_bbox, height=HEIGHT, width=WIDTH)

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
