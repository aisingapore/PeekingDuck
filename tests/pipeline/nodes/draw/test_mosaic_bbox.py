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
Test for draw mosaic_bbox node
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from peekingduck.pipeline.nodes.draw.mosaic_bbox import Node

TEST_IMAGE = ["t1.jpg"]
# path to reach 4 file levels up from test_mosaic_bbox.py
PKD_DIR = Path(__file__).resolve().parents[3]


@pytest.fixture(params=TEST_IMAGE)
def test_image(request):
    test_img_dir = PKD_DIR.parent / "tests" / "data" / "images"

    yield test_img_dir / request.param


@pytest.fixture
def draw_mosaic_node():
    node = Node({"input": ["img", "bboxes"], "output": ["img"], "mosaic_level": 7})
    return node


class TestMosaic:
    def test_no_bbox(self, draw_mosaic_node, test_image):
        original_img = cv2.imread(str(test_image))
        output_img = original_img.copy()

        input = {"img": output_img, "bboxes": []}

        draw_mosaic_node.run(input)
        np.testing.assert_equal(original_img, output_img)

    def test_single_bbox(self, draw_mosaic_node, test_image):
        original_img = cv2.imread(str(test_image))
        frame_height = original_img.shape[0]
        frame_width = original_img.shape[1]
        x1, x2 = int(0.4 * frame_width), int(0.6 * frame_width)
        y1, y2 = int(0.6 * frame_height), int(0.7 * frame_height)
        original_bbox_bounded_area = original_img[y1:y2, x1:x2, :]

        output_img = original_img.copy()
        input = {
            # x1,y1,x2,y2
            "bboxes": [np.asarray([0.4, 0.6, 0.6, 0.7])],
            "img": output_img,
        }

        draw_mosaic_node.run(input)
        output_bbox_bounded_area = output_img[y1:y2, x1:x2, :]

        # does not fail if the area of image are different
        # after applying blurring
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            original_bbox_bounded_area,
            output_bbox_bounded_area,
        )

    def test_multiple_bbox(self, draw_mosaic_node, test_image):
        original_img = cv2.imread(str(test_image))
        frame_height = original_img.shape[0]
        frame_width = original_img.shape[1]

        output_img = original_img.copy()

        input = {
            # x1,y1,x2,y2
            "bboxes": [
                np.asarray([0.4, 0.6, 0.6, 0.7]),
                np.asarray([0.2, 0.1, 0.4, 0.2]),
            ],
            "img": output_img,
        }
        draw_mosaic_node.run(input)
        for bbox in input["bboxes"]:
            x1, y1, x2, y2 = bbox
            x1, x2 = int(x1 * frame_width), int(x2 * frame_width)
            y1, y2 = int(y1 * frame_height), int(y2 * frame_height)

            original_bbox_bounded_area = original_img[y1:y2, x1:x2, :]
            output_bbox_bounded_area = output_img[y1:y2, x1:x2, :]

            # test each bbox the area is the pixel values diff
            # does not fail if the area of image are different
            # after applying blurring
            np.testing.assert_raises(
                AssertionError,
                np.testing.assert_array_equal,
                original_bbox_bounded_area,
                output_bbox_bounded_area,
            )
