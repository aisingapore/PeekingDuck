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

import cv2
import numpy as np

import pytest

from peekingduck.nodes.draw.bbox import Node
from peekingduck.nodes.draw.utils.general import (
    get_image_size,
    project_points_onto_original_image,
)
from tests.conftest import TEST_IMAGES_DIR

BLACK_IMAGE = str(TEST_IMAGES_DIR / "black.jpg")
TURQUOISE = [229, 255, 0]


@pytest.fixture
def draw_bbox_no_labels():
    node = Node(
        {
            "input": ["bboxes", "img", "bbox_labels"],
            "optional_inputs": ["bbox_scores"],
            "output": ["none"],
            "show_labels": False,
            "show_scores": False,
        }
    )
    return node


@pytest.fixture
def draw_bbox_show_labels():
    node = Node(
        {
            "input": ["bboxes", "img", "bbox_labels"],
            "optional_inputs": ["bbox_scores"],
            "output": ["none"],
            "show_labels": True,
            "show_scores": False,
        }
    )
    return node


@pytest.fixture
def draw_bbox_show_scores():
    node = Node(
        {
            "input": ["bboxes", "img", "bbox_labels"],
            "optional_inputs": ["bbox_scores"],
            "output": ["none"],
            "show_labels": False,
            "show_scores": True,
        }
    )
    return node


@pytest.fixture
def draw_bbox_color_choice():
    node = Node(
        {
            "input": ["bboxes", "img", "bbox_labels"],
            "optional_inputs": ["bbox_scores"],
            "output": ["none"],
            "show_labels": False,
            "show_scores": False,
            "color_choice": TURQUOISE,
        }
    )
    return node


@pytest.fixture
def params():
    return {
        "no_bboxes": [],
        "bboxes": [np.array([0.3, 0.2, 0.7, 0.8])],
        "no_labels": [],
        "labels": ["Person"],
        "no_scores": [],
        "scores": [0.99],
    }


class TestBbox:
    """This testing class tests and ensures the following:
    1. The image remains identical after the drawing node without bounding boxes
    2. The bounding box can be placed at the correct position
    3. The labels and scores can be placed at the correct position
    4. The color of the bounding box can be changed correctly
    """

    def test_no_bbox(self, draw_bbox_no_labels, create_image, params):
        original_img = create_image((28, 28, 3))
        output_img = original_img.copy()

        input = {
            "img": output_img,
            "bboxes": params["no_bboxes"],
            "bbox_labels": params["no_labels"],
            "bbox_scores": params["no_scores"],
        }
        draw_bbox_no_labels.run(input)
        # make sure no change to the original image
        np.testing.assert_equal(original_img, output_img)

    def test_bbox(self, draw_bbox_no_labels, params):
        original_img = cv2.imread(BLACK_IMAGE)
        output_img = original_img.copy()
        image_size = get_image_size(original_img)  # (width, height)

        input = {
            "img": output_img,
            "bboxes": params["bboxes"],
            "bbox_labels": params["labels"],  # label comes with bbox
            "bbox_scores": params["no_scores"],
        }
        # get the coordinates for the four corners
        top_left, bottom_right = project_points_onto_original_image(
            input["bboxes"][0], image_size
        )
        draw_bbox_no_labels.run(input)

        # make sure the shape is unchanged
        assert original_img.shape == output_img.shape
        # make sure the image is modified
        np.testing.assert_raises(
            AssertionError, np.testing.assert_equal, original_img, output_img
        )
        # make sure the two bbox corner pixels are changed to the bbox color
        np.testing.assert_equal(
            output_img[top_left[1]][top_left[0]],
            np.array([156, 223, 244]),
        )
        np.testing.assert_equal(
            output_img[bottom_right[1]][bottom_right[0]],
            np.array([156, 223, 244]),
        )

    def test_bbox_and_labels(self, draw_bbox_show_labels, params):
        original_img = cv2.imread(BLACK_IMAGE)
        output_img = original_img.copy()
        image_size = get_image_size(original_img)

        input = {
            "img": output_img,
            "bboxes": params["bboxes"],
            "bbox_labels": params["labels"],
            "bbox_scores": params["no_scores"],
        }
        top_left, bottom_right = project_points_onto_original_image(
            input["bboxes"][0], image_size
        )
        draw_bbox_show_labels.run(input)

        assert original_img.shape == output_img.shape
        np.testing.assert_raises(
            AssertionError, np.testing.assert_equal, original_img, output_img
        )
        np.testing.assert_equal(
            output_img[top_left[1]][top_left[0]],
            np.array([156, 223, 244]),
        )
        np.testing.assert_equal(
            output_img[bottom_right[1]][bottom_right[0]],
            np.array([156, 223, 244]),
        )
        # make sure one pixel of the label box is also modified
        # using the height of the label box
        np.testing.assert_equal(
            output_img[top_left[1] - 32][top_left[0] + 16],
            np.array([156, 223, 244]),
        )

    def test_bbox_and_scores(self, draw_bbox_show_scores, params):
        original_img = cv2.imread(BLACK_IMAGE)
        output_img = original_img.copy()
        image_size = get_image_size(original_img)

        input = {
            "img": output_img,
            "bboxes": params["bboxes"],
            "bbox_labels": params["labels"],
            "bbox_scores": params["scores"],
        }
        top_left, bottom_right = project_points_onto_original_image(
            input["bboxes"][0], image_size
        )
        bottom_left = (top_left[0], bottom_right[1])
        draw_bbox_show_scores.run(input)

        assert original_img.shape == output_img.shape
        np.testing.assert_raises(
            AssertionError, np.testing.assert_equal, original_img, output_img
        )
        np.testing.assert_equal(
            output_img[top_left[1]][top_left[0]],
            np.array([156, 223, 244]),
        )
        np.testing.assert_equal(
            output_img[bottom_right[1]][bottom_right[0]],
            np.array([156, 223, 244]),
        )
        # make sure one pixel of the score box is also modified
        # using the height of the score box
        np.testing.assert_equal(
            output_img[bottom_left[1] - 32][bottom_left[0] + 16],
            np.array([156, 223, 244]),
        )

    def test_bbox_color_choice(self, draw_bbox_color_choice, params):
        """test selected bbox color"""
        original_img = cv2.imread(BLACK_IMAGE)
        output_img = original_img.copy()
        image_size = get_image_size(original_img)

        input = {
            "img": output_img,
            "bboxes": params["bboxes"],
            "bbox_labels": params["labels"],
            "bbox_scores": params["no_scores"],
        }

        top_left, bottom_right = project_points_onto_original_image(
            input["bboxes"][0], image_size
        )
        draw_bbox_color_choice.run(input)

        assert original_img.shape == output_img.shape
        np.testing.assert_raises(
            AssertionError, np.testing.assert_equal, original_img, output_img
        )
        # make sure the two bbox corner pixels are changed to the new bbox color
        np.testing.assert_equal(
            output_img[top_left[1]][top_left[0]],
            np.array(TURQUOISE),
        )
        np.testing.assert_equal(
            output_img[bottom_right[1]][bottom_right[0]],
            np.array(TURQUOISE),
        )
