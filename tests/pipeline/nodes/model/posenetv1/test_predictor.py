# Copyright 2021 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import yaml

from peekingduck.pipeline.nodes.model.posenet import Node
from peekingduck.pipeline.nodes.model.posenetv1.posenet_files.predictor import Predictor

TEST_DIR = Path.cwd() / "images" / "testing"


@pytest.fixture
def posenet_config():
    with open(Path(__file__).resolve().parent / "test_posenet.yml") as file:
        node_config = yaml.safe_load(file)
    node_config["root"] = Path.cwd()
    return node_config


@pytest.fixture
def model_dir(posenet_config):
    return (
        posenet_config["root"].parent
        / "peekingduck_weights"
        / posenet_config["weights"]["model_subdir"]
    )


@pytest.mark.mlmodel
class TestPredictor:
    def test_predictor(self, posenet_config, model_dir):
        predictor = Predictor(posenet_config, model_dir)
        assert predictor is not None, "Predictor is not instantiated"

    def test_predict(self, posenet_config, model_dir):
        frame = cv2.imread(str(TEST_DIR / "t2.jpg"))
        predictor = Predictor(posenet_config, model_dir)
        output = predictor.predict(frame)
        assert len(output) == 4, "Predicted output has missing keys"
        for i in output:
            assert len(i) == 1, "Unexpected number of outputs"

    def test_get_resolution_as_tuple(self, posenet_config, model_dir):
        resolution = {"height": 225, "width": 225}
        predictor = Predictor(posenet_config, model_dir)
        tuple_res = predictor.get_resolution_as_tuple(resolution)
        assert type(tuple_res) is tuple, "Loaded data must be a tuple"
        assert len(tuple_res) == 2, "Loaded data must be of length 2"

    def test_create_image_from_frame(self, posenet_config, model_dir):
        frame = cv2.imread(str(TEST_DIR / "t2.jpg"))
        predictor = Predictor(posenet_config, model_dir)
        _, output_scale, image_size = predictor._create_image_from_frame(
            16, frame, (225, 225), "75"
        )
        assert type(image_size) is list, "Image size must be a list"
        assert image_size == [439, 640], "Incorrect image size"
        assert type(output_scale) is np.ndarray, "Output scale must be a numpy array"
        npt.assert_almost_equal(
            output_scale, np.array([1.95, 2.84]), 2
        ), "Incorrect scale"

    def test_model_instantiation(self, posenet_config, model_dir):
        predictor = Predictor(posenet_config, model_dir)
        posenet_model = predictor._create_posenet_model()
        assert posenet_model is not None, "Model is not instantiated"

    def test_predict_all_poses(self, posenet_config, model_dir):
        frame = cv2.imread(str(TEST_DIR / "t2.jpg"))
        predictor = Predictor(posenet_config, model_dir)
        posenet_model = predictor._create_posenet_model()
        assert posenet_model is not None, "Model is not created"
        coords, scores, masks = predictor._predict_all_poses(posenet_model, frame, "75")
        assert coords.shape == (1, 17, 2), "Coordinates is of wrong shape"
        assert scores.shape == (1, 17), "Scores is of wrong shape"
        assert masks.shape == (1, 17), "Masks is of wrong shape"
