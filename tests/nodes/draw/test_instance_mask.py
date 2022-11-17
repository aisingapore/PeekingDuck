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
Tests for draw instance mask node.
"""

import cv2
import numpy as np
import pytest
import yaml
from skimage.measure import compare_ssim as ssim

from peekingduck.nodes.draw.instance_mask import Node
from tests.conftest import PKD_DIR, TEST_DATA_DIR, TEST_IMAGES_DIR

IMAGE_ORIGINAL = "draw_instance_mask_original_image.jpg"
IMAGE_UNCHANGED_AFTER_PIPELINE = "draw_instance_mask_image_unchanged_after_pipeline.jpg"
IMAGE_WITH_MASKS = "draw_instance_mask_image_with_masks.jpg"
IMAGE_WITH_CONTOURED_MASKS = "draw_instance_mask_image_with_contoured_masks.jpg"
IMAGE_WITH_BLUR_EFFECT = "draw_instance_mask_image_with_blur_effect.jpg"
IMAGE_WITH_MOSAIC_EFFECT = "draw_instance_mask_image_with_mosaic_effect.jpg"
IMAGE_WITH_BLUR_EFFECT_BACKGROUND_AREA = (
    "draw_instance_mask_image_with_blur_effect_background_area.jpg"
)
IMAGE_ADJUSTED_CONTRAST = "draw_instance_mask_image_adjusted_contrast.jpg"
IMAGE_ADJUSTED_BRIGHTNESS = "draw_instance_mask_image_adjusted_brightness.jpg"
IMAGE_GAMMA_CORRECTION = "draw_instance_mask_image_gamma_correction.jpg"
TEST_DATA_SUBDIR = "instance_mask"
INPUTS_NPZ = "draw_instance_mask_inputs.npz"
SSIM_THRESHOLD = 0.97


@pytest.fixture()
def image_original():
    return TEST_IMAGES_DIR / IMAGE_ORIGINAL


@pytest.fixture()
def image_unchanged_after_pipeline():
    return TEST_IMAGES_DIR / IMAGE_UNCHANGED_AFTER_PIPELINE


@pytest.fixture()
def image_with_masks():
    return TEST_IMAGES_DIR / IMAGE_WITH_MASKS


@pytest.fixture()
def image_with_contoured_masks():
    return TEST_IMAGES_DIR / IMAGE_WITH_CONTOURED_MASKS


@pytest.fixture()
def image_with_blur_effect():
    return TEST_IMAGES_DIR / IMAGE_WITH_BLUR_EFFECT


@pytest.fixture()
def image_with_mosaic_effect():
    return TEST_IMAGES_DIR / IMAGE_WITH_MOSAIC_EFFECT


@pytest.fixture()
def image_with_blur_effect_background_area():
    return TEST_IMAGES_DIR / IMAGE_WITH_BLUR_EFFECT_BACKGROUND_AREA


@pytest.fixture()
def image_adjusted_contrast():
    return TEST_IMAGES_DIR / IMAGE_ADJUSTED_CONTRAST


@pytest.fixture()
def image_adjusted_brightness():
    return TEST_IMAGES_DIR / IMAGE_ADJUSTED_BRIGHTNESS


@pytest.fixture()
def image_gamma_correction():
    return TEST_IMAGES_DIR / IMAGE_GAMMA_CORRECTION


@pytest.fixture()
def draw_mask_inputs():
    """Returns dictionary of masks, bbox_labels, bbox_scores."""
    inputs = dict(np.load(TEST_DATA_DIR / TEST_DATA_SUBDIR / INPUTS_NPZ))

    return inputs


@pytest.fixture
def draw_instance_mask_config():
    with open(PKD_DIR / "configs" / "draw" / "instance_mask.yml") as infile:
        node_config = yaml.safe_load(infile)

    return node_config


@pytest.fixture(
    params=[
        {"key": "instance_color_scheme", "value": "no_such_scheme"},
        {"key": "effect_area", "value": "no_such_effect_area"},
    ],
)
def draw_instance_mask_bad_config_values(request, draw_instance_mask_config):
    draw_instance_mask_config[request.param["key"]] = request.param["value"]
    return draw_instance_mask_config


@pytest.fixture(
    params=[
        {
            "key": "effect",
            "value": {
                "contrast": None,
                "brightness": None,
                "gamma_correction": None,
                "blur": 50,
                "mosaic": 25,
            },
        },
    ],
)
def draw_instance_mask_bad_number_of_config_values(request, draw_instance_mask_config):
    draw_instance_mask_config[request.param["key"]] = request.param["value"]
    return draw_instance_mask_config


@pytest.fixture(
    params=[
        {
            "key": "contrast",
            "value": "yes",
        },
        {
            "key": "contrast",
            "value": -0.1,
        },
        {
            "key": "contrast",
            "value": 3.1,
        },
        {
            "key": "brightness",
            "value": "yes",
        },
        {
            "key": "brightness",
            "value": -100.1,
        },
        {
            "key": "brightness",
            "value": 100.1,
        },
        {
            "key": "gamma_correction",
            "value": "yes",
        },
        {
            "key": "gamma_correction",
            "value": -0.1,
        },
        {
            "key": "blur",
            "value": "yes",
        },
        {
            "key": "blur",
            "value": 0,
        },
        {
            "key": "mosaic",
            "value": "yes",
        },
        {
            "key": "mosaic",
            "value": 0,
        },
    ],
)
def draw_instance_mask_bad_effect_config_values(request, draw_instance_mask_config):
    draw_instance_mask_config["effect"][request.param["key"]] = request.param["value"]
    return draw_instance_mask_config


@pytest.fixture(
    params=[
        {
            "key": "show",
            "value": "yes",
        },
        {
            "key": "thickness",
            "value": "yes",
        },
        {
            "key": "thickness",
            "value": 0.9,
        },
    ],
)
def draw_instance_mask_bad_contours_config_values(request, draw_instance_mask_config):
    draw_instance_mask_config["contours"][request.param["key"]] = request.param["value"]
    return draw_instance_mask_config


@pytest.fixture
def draw_standard_instance_mask_node(draw_instance_mask_config):
    return Node(draw_instance_mask_config)


@pytest.fixture
def draw_standard_instance_mask_node_with_contours(draw_instance_mask_config):
    draw_instance_mask_config["contours"]["show"] = True
    draw_instance_mask_config["contours"]["thickness"] = 2

    return Node(draw_instance_mask_config)


@pytest.fixture
def draw_instance_mask_node_with_blur_effect(draw_instance_mask_config):
    draw_instance_mask_config["effect"]["blur"] = 50

    return Node(draw_instance_mask_config)


@pytest.fixture
def draw_instance_mask_node_with_mosaic_effect(draw_instance_mask_config):
    draw_instance_mask_config["effect"]["mosaic"] = 25

    return Node(draw_instance_mask_config)


@pytest.fixture
def draw_instance_mask_node_with_blur_effect_background_area(draw_instance_mask_config):
    draw_instance_mask_config["effect"]["blur"] = 50
    draw_instance_mask_config["effect_area"] = "background"

    return Node(draw_instance_mask_config)


@pytest.fixture
def draw_instance_mask_node_adjust_contrast(draw_instance_mask_config):
    draw_instance_mask_config["effect"]["contrast"] = 1.2

    return Node(draw_instance_mask_config)


@pytest.fixture
def draw_instance_mask_node_adjust_brightness(draw_instance_mask_config):
    draw_instance_mask_config["effect"]["brightness"] = 20

    return Node(draw_instance_mask_config)


@pytest.fixture
def draw_instance_mask_node_gamma_correction(draw_instance_mask_config):
    draw_instance_mask_config["effect"]["gamma_correction"] = 0.8

    return Node(draw_instance_mask_config)


class TestDrawInstanceMasks:
    def test_invalid_config_values(self, draw_instance_mask_bad_config_values):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=draw_instance_mask_bad_config_values)
        assert "must be" in str(excinfo.value)

    def test_invalid_number_of_effect_config_values(
        self, draw_instance_mask_bad_number_of_config_values
    ):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=draw_instance_mask_bad_number_of_config_values)
        assert "can be" in str(excinfo.value)

    def test_invalid_effect_config_values(
        self, draw_instance_mask_bad_effect_config_values
    ):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=draw_instance_mask_bad_effect_config_values)
        assert "must be" in str(excinfo.value)

    def test_invalid_contours_config_values(
        self, draw_instance_mask_bad_contours_config_values
    ):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=draw_instance_mask_bad_contours_config_values)
        assert "must be" in str(excinfo.value)

    def test_no_labels(
        self,
        draw_standard_instance_mask_node,
        draw_mask_inputs,
        image_original,
        image_unchanged_after_pipeline,
    ):
        original_img = cv2.imread(str(image_original))
        output_img = original_img.copy()
        draw_mask_inputs["img"] = output_img
        draw_mask_inputs["bbox_labels"] = []
        outputs = draw_standard_instance_mask_node.run(draw_mask_inputs)

        assert TestDrawInstanceMasks._image_equal_with_ground_truth_jpeg(
            outputs["img"], image_unchanged_after_pipeline
        )

    # fmt: off
    @pytest.mark.parametrize(
        "pkd_node, ground_truth_image_path",
        [
            (pytest.lazy_fixture(("draw_standard_instance_mask_node", "image_with_masks"))),
            (pytest.lazy_fixture(("draw_standard_instance_mask_node_with_contours", "image_with_contoured_masks"))),
            (pytest.lazy_fixture(("draw_instance_mask_node_with_blur_effect", "image_with_blur_effect"))),
            (pytest.lazy_fixture(("draw_instance_mask_node_with_mosaic_effect", "image_with_mosaic_effect"))),
            (pytest.lazy_fixture(("draw_instance_mask_node_with_blur_effect_background_area", "image_with_blur_effect_background_area"))),
            (pytest.lazy_fixture(("draw_instance_mask_node_adjust_contrast", "image_adjusted_contrast"))),
            (pytest.lazy_fixture(("draw_instance_mask_node_adjust_brightness", "image_adjusted_brightness"))),
            (pytest.lazy_fixture(("draw_instance_mask_node_gamma_correction", "image_gamma_correction"))),
        ],
    )
    # fmt: on
    def test_masks(
        self,
        pkd_node,
        draw_mask_inputs,
        image_original,
        ground_truth_image_path,
    ):
        original_img = cv2.imread(str(image_original))
        inputs = draw_mask_inputs
        inputs["img"] = original_img
        outputs = pkd_node.run(inputs)

        assert TestDrawInstanceMasks._image_equal_with_ground_truth_jpeg(
            outputs["img"], ground_truth_image_path
        )

    @staticmethod
    def _image_equal_with_ground_truth_jpeg(
        output_image: np.ndarray, ground_truth_jpeg_path: str
    ) -> bool:
        ground_truth_image = cv2.imread(str(ground_truth_jpeg_path))

        return (
            ssim(output_image, ground_truth_image, multichannel=True) > SSIM_THRESHOLD
        )
