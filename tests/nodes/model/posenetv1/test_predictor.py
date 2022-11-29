# Copyright 2022 AI Singapore
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

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import yaml

from peekingduck.nodes.model.posenetv1.posenet_files.predictor import Predictor
from tests.conftest import PKD_DIR


@pytest.fixture
def posenet_config():
    with open(PKD_DIR / "configs" / "model" / "posenet.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = PKD_DIR
    # Only test model_type=75 instead of the default resnet or other types.
    node_config["model_type"] = 75

    return node_config


@pytest.fixture
def model_dir(posenet_config):
    return (
        posenet_config["root"].parent
        / "peekingduck_weights"
        / posenet_config["weights"][posenet_config["model_format"]]["model_subdir"]
        / posenet_config["model_format"]
    )


@pytest.mark.mlmodel
class TestPredictor:
    def test_predictor(self, posenet_config, model_dir):
        predictor = Predictor(
            model_dir,
            posenet_config["model_type"],
            posenet_config["weights"][posenet_config["model_format"]]["model_file"],
            posenet_config["model_nodes"],
            posenet_config["resolution"],
            posenet_config["max_pose_detection"],
            posenet_config["score_threshold"],
        )
        assert predictor is not None, "Predictor is not instantiated"

    def test_predict(self, single_person_image, posenet_config, model_dir):
        single_person_img = cv2.imread(single_person_image)
        predictor = Predictor(
            model_dir,
            posenet_config["model_type"],
            posenet_config["weights"][posenet_config["model_format"]]["model_file"],
            posenet_config["model_nodes"],
            posenet_config["resolution"],
            posenet_config["max_pose_detection"],
            posenet_config["score_threshold"],
        )
        output = predictor.predict(single_person_img)
        assert len(output) == 4, "Predicted output has missing keys"
        for i in output:
            assert len(i) == 1, "Unexpected number of outputs"

    def test_preprocess(self, single_person_image, posenet_config, model_dir):
        single_person_img = cv2.imread(single_person_image)
        predictor = Predictor(
            model_dir,
            posenet_config["model_type"],
            posenet_config["weights"][posenet_config["model_format"]]["model_file"],
            posenet_config["model_nodes"],
            posenet_config["resolution"],
            posenet_config["max_pose_detection"],
            posenet_config["score_threshold"],
        )
        _, output_scale, image_size = predictor._preprocess(single_person_img)
        assert isinstance(image_size, list), "Image size must be a list"
        assert image_size == [439, 640], "Incorrect image size"
        assert isinstance(output_scale, list), "Output scale must be a numpy array"
        npt.assert_almost_equal(
            output_scale, np.array([1.95, 2.84]), 2, err_msg="Incorrect scale"
        )

    def test_model_instantiation(self, posenet_config, model_dir):
        predictor = Predictor(
            model_dir,
            posenet_config["model_type"],
            posenet_config["weights"][posenet_config["model_format"]]["model_file"],
            posenet_config["model_nodes"],
            posenet_config["resolution"],
            posenet_config["max_pose_detection"],
            posenet_config["score_threshold"],
        )
        posenet_model = predictor._create_posenet_model()
        assert posenet_model is not None, "Model is not instantiated"

    def test_predict_all_poses(self, single_person_image, posenet_config, model_dir):
        single_person_img = cv2.imread(single_person_image)
        predictor = Predictor(
            model_dir,
            posenet_config["model_type"],
            posenet_config["weights"][posenet_config["model_format"]]["model_file"],
            posenet_config["model_nodes"],
            posenet_config["resolution"],
            posenet_config["max_pose_detection"],
            posenet_config["score_threshold"],
        )
        posenet_model = predictor._create_posenet_model()
        assert posenet_model is not None, "Model is not created"
        coords, scores = predictor._predict_all_poses(single_person_img)
        assert coords.shape == (1, 17, 2), "Coordinates is of wrong shape"
        assert scores.shape == (1, 17), "Scores is of wrong shape"

    def test_get_bbox_of_one_pose(self, single_person_image, posenet_config, model_dir):
        single_person_img = cv2.imread(single_person_image)
        predictor = Predictor(
            model_dir,
            posenet_config["model_type"],
            posenet_config["weights"][posenet_config["model_format"]]["model_file"],
            posenet_config["model_nodes"],
            posenet_config["resolution"],
            posenet_config["max_pose_detection"],
            posenet_config["score_threshold"],
        )
        posenet_model = predictor._create_posenet_model()
        assert posenet_model is not None, "Model is not created"
        coords, scores = predictor._predict_all_poses(single_person_img)
        assert coords.shape == (1, 17, 2), "Coordinates is of wrong shape"
        assert scores.shape == (1, 17), "Scores is of wrong shape"
