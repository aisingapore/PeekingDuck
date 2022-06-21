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

import tempfile
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest
import yaml
from pytest_lazyfixture import lazy_fixture

from peekingduck.pipeline.nodes.draw.instance_mask import Node
from tests.conftest import PKD_DIR, TEST_DATA_DIR, TEST_IMAGES_DIR

IMAGE_ORIGINAL = ["draw_instance_mask_original_image.jpg"]
IMAGE_UNCHANGED_AFTER_PIPELINE = [
    "draw_instance_mask_image_unchanged_after_pipeline.jpg"
]
IMAGE_WITH_MASKS = ["draw_instance_mask_image_with_masks.jpg"]
IMAGE_WITH_CONTOURED_MASKS = ["draw_instance_mask_image_with_contoured_masks.jpg"]
IMAGE_WITH_BLUR_EFFECT = ["draw_instance_mask_image_with_blur_effect.jpg"]
IMAGE_WITH_MOSAIC_EFFECT = ["draw_instance_mask_image_with_mosaic_effect.jpg"]
IMAGE_WITH_BLUR_EFFECT_UNMMASKED_AREA = [
    "draw_instance_mask_image_with_blur_effect_unmasked_area.jpg"
]
IMAGE_ADJUSTED_CONTRAST_BRIGHTNESS = [
    "draw_instance_mask_image-adjusted-contrast-brightness.jpg"
]
IMAGE_GAMMA_CORRECTION = ["draw_instance_mask_image-gamma-correction.jpg"]
###############################################################################
# Remember to change the INPUTS_SUBDIR!
###############################################################################
INPUTS_SUBDIR = "node_inputs"
INPUTS_NPZ = "draw_instance_mask_inputs.npz"


@pytest.fixture(params=IMAGE_ORIGINAL)
def image_original(request):
    return TEST_IMAGES_DIR / request.param


@pytest.fixture(params=IMAGE_UNCHANGED_AFTER_PIPELINE)
def image_unchanged_after_pipeline(request):
    return TEST_IMAGES_DIR / request.param


@pytest.fixture(params=IMAGE_WITH_MASKS)
def image_with_masks(request):
    return TEST_IMAGES_DIR / request.param


@pytest.fixture(params=IMAGE_WITH_CONTOURED_MASKS)
def image_with_contoured_masks(request):
    return TEST_IMAGES_DIR / request.param


@pytest.fixture(params=IMAGE_WITH_BLUR_EFFECT)
def image_with_blur_effect(request):
    return TEST_IMAGES_DIR / request.param


@pytest.fixture(params=IMAGE_WITH_MOSAIC_EFFECT)
def image_with_mosaic_effect(request):
    return TEST_IMAGES_DIR / request.param


@pytest.fixture(params=IMAGE_WITH_BLUR_EFFECT_UNMMASKED_AREA)
def image_with_blur_effect_unmasked_area(request):
    return TEST_IMAGES_DIR / request.param


@pytest.fixture(params=IMAGE_ADJUSTED_CONTRAST_BRIGHTNESS)
def image_adjusted_contrast_brightness(request):
    return TEST_IMAGES_DIR / request.param


@pytest.fixture(params=IMAGE_GAMMA_CORRECTION)
def image_gamma_correction(request):
    return TEST_IMAGES_DIR / request.param


@pytest.fixture()
def draw_mask_inputs():
    """Returns dictionary of masks, bbox_labels, bbox_scores."""
    inputs = dict(np.load(TEST_DATA_DIR / INPUTS_SUBDIR / INPUTS_NPZ))

    return inputs


@pytest.fixture
def draw_instance_mask_config():
    with open(PKD_DIR / "configs" / "draw" / "instance_mask.yml") as infile:
        node_config = yaml.safe_load(infile)

    return node_config


@pytest.fixture(
    params=[
        {"key": "instance_color_scheme", "value": "no_such_scheme"},
        {"key": "effect", "value": "no_such_effect"},
        {"key": "effect_area", "value": "no_such_effect_area"},
        {"key": "gamma", "value": -0.5},
        {"key": "gamma", "value": "-"},
        {"key": "alpha", "value": -0.5},
        {"key": "alpha", "value": "-"},
        {"key": "beta", "value": 0.5},
        {"key": "beta", "value": 256},
        {"key": "beta", "value": "-"},
        {"key": "blur_kernel_size", "value": 0.5},
        {"key": "blur_kernel_size", "value": "-"},
        {"key": "mosaic_level", "value": 0.5},
        {"key": "mosaic_level", "value": "-"},
        {"key": "show_contours", "value": "yes"},
        {"key": "contour_thickness", "value": 0.5},
        {"key": "contour_thickness", "value": "-"},
    ],
)
def draw_instance_mask_bad_config_value(request, draw_instance_mask_config):
    draw_instance_mask_config[request.param["key"]] = request.param["value"]
    return draw_instance_mask_config


@pytest.fixture
def draw_instance_mask_node(draw_instance_mask_config):
    return Node(draw_instance_mask_config)


@pytest.fixture
def draw_instance_mask_node_with_contours(draw_instance_mask_config):
    draw_instance_mask_config["show_contours"] = True
    draw_instance_mask_config["contour_thickness"] = 3

    return Node(draw_instance_mask_config)


@pytest.fixture
def draw_instance_mask_node_with_blur_effect(draw_instance_mask_config):
    draw_instance_mask_config["effect"] = "blur"
    draw_instance_mask_config["blur_kernel_size"] = 50

    return Node(draw_instance_mask_config)


@pytest.fixture
def draw_instance_mask_node_with_mosaic_effect(draw_instance_mask_config):
    draw_instance_mask_config["effect"] = "mosaic"
    draw_instance_mask_config["mosaic_level"] = 25

    return Node(draw_instance_mask_config)


@pytest.fixture
def draw_instance_mask_node_with_blur_effect_unmasked_area(draw_instance_mask_config):
    draw_instance_mask_config["effect"] = "blur"
    draw_instance_mask_config["effect_area"] = "unmasked"
    draw_instance_mask_config["blur_kernel_size"] = 50

    return Node(draw_instance_mask_config)


@pytest.fixture
def draw_instance_mask_node_adjust_contrast_brightness(draw_instance_mask_config):
    draw_instance_mask_config["effect"] = "contrast_brightness"
    draw_instance_mask_config["alpha"] = 1.2
    draw_instance_mask_config["beta"] = 20

    return Node(draw_instance_mask_config)


@pytest.fixture
def draw_instance_mask_node_gamma_correction(draw_instance_mask_config):
    draw_instance_mask_config["effect"] = "gamma_correction"
    draw_instance_mask_config["gamma"] = 0.8

    return Node(draw_instance_mask_config)


class TestDrawInstanceMasks:
    def test_invalid_config_value(self, draw_instance_mask_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=draw_instance_mask_bad_config_value)
        assert "must be" in str(excinfo.value)

    def test_no_scores(
        self,
        draw_instance_mask_node,
        draw_mask_inputs,
        image_original,
        image_unchanged_after_pipeline,
    ):
        original_img = cv.imread(str(image_original))
        output_img = original_img.copy()
        draw_mask_inputs["img"] = output_img
        draw_mask_inputs["bbox_scores"] = []
        outputs = draw_instance_mask_node.run(draw_mask_inputs)

        assert TestDrawInstanceMasks._image_equal_with_ground_truth_jpeg(
            outputs["img"], image_unchanged_after_pipeline
        )

    # fmt: off
    @pytest.mark.parametrize(
        'pkd_node, ground_truth_image',
        [
            (lazy_fixture(('draw_instance_mask_node', 'image_with_masks'))),
            (lazy_fixture(('draw_instance_mask_node_with_contours', 'image_with_contoured_masks'))),
            (lazy_fixture(('draw_instance_mask_node_with_blur_effect', 'image_with_blur_effect'))),
            (lazy_fixture(('draw_instance_mask_node_with_mosaic_effect', 'image_with_mosaic_effect'))),
            (lazy_fixture(('draw_instance_mask_node_with_blur_effect_unmasked_area', 'image_with_blur_effect_unmasked_area'))),
            (lazy_fixture(('draw_instance_mask_node_adjust_contrast_brightness', 'image_adjusted_contrast_brightness'))),
            (lazy_fixture(('draw_instance_mask_node_gamma_correction', 'image_gamma_correction'))),
        ],
    )
    # fmt: on
    def test_masks(
        self,
        pkd_node,
        draw_mask_inputs,
        image_original,
        ground_truth_image,
    ):
        original_img = cv.imread(str(image_original))
        inputs = draw_mask_inputs
        inputs["img"] = original_img
        outputs = pkd_node.run(inputs)

        assert TestDrawInstanceMasks._image_equal_with_ground_truth_jpeg(
            outputs["img"], ground_truth_image
        )

    @staticmethod
    def _image_equal_with_ground_truth_jpeg(
        output_image: np.ndarray, ground_truth_jpeg_path: str
    ) -> bool:
        with tempfile.NamedTemporaryFile(suffix=".jpg") as fp:
            cv.imwrite(fp.name, output_image)

            return (
                Path(fp.name).stat().st_size
                == Path(ground_truth_jpeg_path).stat().st_size
                and fp.read() == open(ground_truth_jpeg_path, "rb").read()
            )
