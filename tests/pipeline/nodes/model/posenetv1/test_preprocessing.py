"""
Copyright 2021 AI Singapore

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pathlib import Path

import cv2
import numpy as np
import numpy.testing as npt
import pytest

from peekingduck.pipeline.nodes.model.posenetv1.posenet_files.preprocessing import (
    _get_valid_resolution,
    _rescale_image,
    rescale_image,
)

TEST_DIR = Path.cwd() / "images" / "testing"
NP_FILE = np.load(Path(__file__).resolve().parent / "posenet.npz")
MOBILENET_MODELS = [50, 75, 100]


@pytest.fixture
def frame():
    image = cv2.imread(str(TEST_DIR / "t1.jpg"))
    return np.array(image)


@pytest.fixture
def image_resnet():
    return NP_FILE["image_resnet"]


@pytest.fixture
def image_mobilenet():
    return NP_FILE["image"]


@pytest.fixture
def mobilenet_model(request):
    yield request.param


class TestPreprocessing:
    def test_resize_image(self, frame):
        image_processed, scale = rescale_image(frame, (500, 333), 1.5, 16, "resnet")
        assert image_processed.shape == (
            1,
            497,
            737,
            3,
        ), "Rescaled image is of incorrect shape"
        npt.assert_almost_equal(scale, np.array([0.8684, 0.8551]), 4), "Incorrect scale"

    def test_preprocess_image_resnet(self, frame, image_resnet):
        image_processed, scale = rescale_image(frame, (225, 225), 1, 16, "resnet")
        assert image_processed.shape == (
            1,
            225,
            225,
            3,
        ), "Rescaled image is of incorrect shape"
        assert scale == pytest.approx(np.array([2.844, 1.888]), 0.01), "Incorrect scale"
        npt.assert_almost_equal(
            image_processed, image_resnet, 2
        ), "Processed resnet image did not meet expected value"

    @pytest.mark.parametrize(
        "mobilenet_model", MOBILENET_MODELS, indirect=True, ids=str
    )
    def test_preprocess_image_mobilenet(self, frame, image_mobilenet, mobilenet_model):
        image_processed, scale = rescale_image(
            frame, (225, 225), 1, 16, mobilenet_model
        )
        assert image_processed.shape == (
            1,
            225,
            225,
            3,
        ), "Rescaled image is of incorrect shape"
        assert scale == pytest.approx(np.array([2.844, 1.888]), 0.01), "Incorrect scale"
        npt.assert_almost_equal(
            image_processed, image_mobilenet, 2
        ), "Processed mobilenet image did not meet expected value"

    def test_get_valid_resolution(self):
        assert _get_valid_resolution(183.0, 183.0, 16) == (
            177,
            177,
        ), "Unable to obtain valid resolution"
        assert _get_valid_resolution(225.0, 225.0, 32) == (
            225,
            225,
        ), "Unable to obtain valid resolution"

    @pytest.mark.parametrize(
        "mobilenet_model", MOBILENET_MODELS, indirect=True, ids=str
    )
    def test_rescale_image_mobilenet(self, frame, image_mobilenet, mobilenet_model):
        rescaled_image = _rescale_image(frame, 225, 225, mobilenet_model)
        assert rescaled_image.shape == (
            1,
            225,
            225,
            3,
        ), "Rescaled image is of incorrect shape"
        npt.assert_almost_equal(
            rescaled_image, image_mobilenet, 2
        ), "Processed mobilenet image did not meet expected value"

    def test_rescale_image_resnet(self, frame, image_resnet):
        rescaled_image = _rescale_image(frame, 225, 225, "resnet")
        assert rescaled_image.shape == (
            1,
            225,
            225,
            3,
        ), "Rescaled image is of incorrect shape"
        npt.assert_almost_equal(
            rescaled_image, image_resnet, 2
        ), "Processed resnet image did not meet expected value"
