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
Test for draw bbox node
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from peekingduck.pipeline.nodes.draw.bbox import Node

BLACK_IMAGE = ["black.jpg"]
# path to reach 4 file levels up from test_bbox.py
PKD_DIR = Path(__file__).resolve().parents[3]


@pytest.fixture(params=BLACK_IMAGE)
def black_image(request):
    test_img_dir = PKD_DIR.parent / "tests" / "data" / "images"

    yield test_img_dir / request.param


@pytest.fixture
def draw_bbox_no_labels():
    node = Node(
        {
            "input": ["bboxes", "img", "bbox_labels"],
            "output": ["none"],
            "show_labels": False,
        }
    )
    return node


@pytest.fixture
def draw_bbox_show_labels():
    node = Node(
        {
            "input": ["bboxes", "img", "bbox_labels"],
            "output": ["none"],
            "show_labels": True,
        }
    )
    return node


class TestBbox:
    def test_no_bbox(self, draw_bbox_no_labels, create_image):
        no_bboxes = []
        original_img = create_image((28, 28, 3))
        output_img = original_img.copy()
        no_labels = []
        input1 = {"bboxes": no_bboxes, "img": output_img, "bbox_labels": no_labels}
        draw_bbox_no_labels.run(input1)
        np.testing.assert_equal(original_img, output_img)

    def test_bbox_no_label(
        self, draw_bbox_no_labels, draw_bbox_show_labels, black_image
    ):
        bboxes = [np.array([0, 0, 1, 1])]
        original_img = cv2.imread(str(black_image))
        output_img_no_label = original_img.copy()
        output_img_show_label = original_img.copy()
        labels = ["Person"]
        input1 = {"bboxes": bboxes, "img": output_img_no_label, "bbox_labels": labels}
        input2 = {"bboxes": bboxes, "img": output_img_show_label, "bbox_labels": labels}
        draw_bbox_no_labels.run(input1)
        # after running draw, should not be equal
        assert original_img.shape == output_img_no_label.shape
        np.testing.assert_raises(
            AssertionError, np.testing.assert_equal, original_img, output_img_no_label
        )
        # assert the top left pixel is replaced with bbox color
        np.testing.assert_equal(output_img_no_label[0][0], np.array([156, 223, 244]))

        # test with labels
        draw_bbox_show_labels.run(input2)
        assert original_img.shape == output_img_show_label.shape
        np.testing.assert_raises(
            AssertionError, np.testing.assert_equal, original_img, output_img_show_label
        )
        np.testing.assert_equal(output_img_no_label[0][0], np.array([156, 223, 244]))
