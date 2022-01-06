# Copyright 2021 AI Singapore
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

import gc
from pathlib import Path

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow.keras.backend as K
import yaml

from peekingduck.pipeline.nodes.model.movenetv1.movenet_files.predictor import Predictor

TEST_DIR = Path.cwd() / "images" / "testing"


@pytest.fixture(params=["t2.jpg"])
def single_person_image(request):
    yield request.param


@pytest.fixture
def movenet_config():
    with open(Path(__file__).resolve().parent / "test_movenet.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = Path.cwd()

    yield node_config
    K.clear_session()
    gc.collect()


@pytest.fixture
def model_dir(movenet_config):
    return (
        movenet_config["root"].parent
        / "peekingduck_weights"
        / movenet_config["weights"]["model_subdir"]
    )


@pytest.mark.mlmodel
class TestPredictor:
    def test_predictor(self, movenet_config, model_dir):
        movenet_predictor = Predictor(movenet_config, model_dir)
        assert movenet_predictor is not None, "Predictor is not instantiated"

    def test_model_creation(self, movenet_config, model_dir):
        movenet_predictor = Predictor(movenet_config, model_dir)
        assert movenet_predictor.movenet_model is not None, "Model is not loaded"

    def test_get_resolution_as_tuple(self, movenet_config, model_dir):
        resolution = {"height": 256, "width": 256}
        movenet_predictor = Predictor(movenet_config, model_dir)
        tuple_res = movenet_predictor.get_resolution_as_tuple(resolution)
        assert isinstance(
            tuple_res, tuple
        ), f"Resolution in config must be a tuple instead of {type(tuple_res)}"
        assert tuple_res == (
            256,
            256,
        ), f"Incorrect resolution: expected (256, 256) but got {tuple_res}"
        assert (
            len(tuple_res) == 2
        ), f"Wrong resolution dimension: expected 2 but got {len(tuple_res)}"

    def test_predict(self, movenet_config, model_dir, single_person_image):
        img = cv2.imread(str(TEST_DIR / single_person_image))
        movenet_predictor = Predictor(movenet_config, model_dir)
        (
            bboxes,
            valid_keypoints,
            keypoints_scores,
            keypoints_conns,
        ) = movenet_predictor.predict(img)
        assert bboxes.shape == (1, 4)
        assert valid_keypoints.shape == (1, 17, 2)
        assert keypoints_scores.shape == (1, 17)
        assert keypoints_conns.shape == (1, 19, 2, 2)

    def test_get_results_single(self, movenet_config, model_dir):
        # prediction for singlepose model is in shape of [1,1,17,13]
        # generates random tensor with values from 0.3 to 0.9
        prediction = tf.random.uniform(
            (1, 1, 17, 3), minval=0.3, maxval=0.9, dtype=tf.dtypes.float32, seed=24
        )
        movenet_predictor = Predictor(movenet_config, model_dir)
        (
            bbox,
            valid_keypoints,
            keypoints_scores,
            keypoints_conns,
        ) = movenet_predictor._get_results_single(prediction)
        assert bbox.shape == (1, 4)
        assert valid_keypoints.shape == (1, 17, 2)
        assert keypoints_scores.shape == (1, 17)
        assert keypoints_conns.shape[0] == 1

        # generates random tensor with values from 0.0 to 0.1
        # since values are below config score threshold
        # predictions should be tuples of empty np array
        prediction_no_pose = tf.random.uniform(
            (1, 1, 17, 3), minval=0.0, maxval=0.1, dtype=tf.dtypes.float32, seed=24
        )
        (
            bbox_no_pose,
            valid_keypoints_no_pose,
            keypoints_scores_no_pose,
            keypoints_conns_no_pose,
        ) = movenet_predictor._get_results_single(prediction_no_pose)
        npt.assert_array_equal(
            x=bbox_no_pose,
            y=np.zeros(0),
            err_msg=(
                "Unexpected bbox output for prediction with keypoint score below "
                f"threshold got {bbox_no_pose} instead of {np.zeros(0)}"
            ),
        )
        npt.assert_array_equal(
            x=valid_keypoints_no_pose,
            y=np.zeros(0),
            err_msg=(
                "Unexpected valid keypoint output for prediction with keypoint score "
                f"below threshold got {valid_keypoints_no_pose} instead of {np.zeros(0)}"
            ),
        )
        npt.assert_array_equal(
            x=keypoints_scores_no_pose,
            y=np.zeros(0),
            err_msg=(
                "Unexpected keypoint score output for prediction with keypoint score "
                f"below threshold got {keypoints_scores_no_pose} instead of {np.zeros(0)}"
            ),
        )
        npt.assert_array_equal(
            x=keypoints_conns_no_pose,
            y=np.zeros(0),
            err_msg=(
                "Unexpected keypoint connection output for prediction with keypoint "
                f"score below threshold got {keypoints_conns_no_pose} instead of {np.zeros(0)}"
            ),
        )

    def test_get_results_multi(self, movenet_config, model_dir):
        # prediction for multi model is in shape of [1,6,56]
        # generates random tensor with values from 0.2 to 0.9
        # since threshold in config is at 0.2, this random tensor
        # will have at least 1 pose after filtering
        prediction = tf.random.uniform(
            (1, 6, 56), minval=0.2, maxval=0.9, dtype=tf.dtypes.float32, seed=24
        )
        movenet_predictor = Predictor(movenet_config, model_dir)
        (
            bbox,
            valid_keypoints,
            keypoints_scores,
            keypoints_conns,
        ) = movenet_predictor._get_results_multi(prediction)
        # output of random tensor will produce between 1 to 6 valid output
        # The valid number of detections will be same for the all outputs,
        # which is the value of the 1st index in the shape of the outputs
        assert bbox.shape[0] >= 1
        assert bbox.shape[0] <= 6
        assert bbox.shape[0] == valid_keypoints.shape[0]
        assert bbox.shape[0] == keypoints_scores.shape[0]
        assert bbox.shape[0] == keypoints_conns.shape[0]

        # generates random tensor with values from 0.0 to 0.1
        # since values are below config score threshold
        # predictions should be tuples of empty numpy arrays
        # but with different shapes
        prediction_no_pose = tf.random.uniform(
            (1, 6, 56), minval=0.0, maxval=0.1, dtype=tf.dtypes.float32, seed=24
        )
        (
            bbox_no_pose,
            valid_keypoints_no_pose,
            keypoints_scores_no_pose,
            keypoints_conns_no_pose,
        ) = movenet_predictor._get_results_multi(prediction_no_pose)
        npt.assert_array_equal(
            x=bbox_no_pose,
            y=np.zeros(0),
            err_msg=(
                "Unexpected bbox output for prediction with keypoint score below "
                f"threshold got {bbox_no_pose} instead of {np.zeros(0)}"
            ),
        )
        npt.assert_array_equal(
            x=valid_keypoints_no_pose,
            y=np.zeros(0),
            err_msg=(
                "Unexpected valid keypoint output for prediction with keypoint score "
                f"below threshold got {valid_keypoints_no_pose} instead of {np.zeros(0)}"
            ),
        )
        npt.assert_array_equal(
            x=keypoints_scores_no_pose,
            y=np.zeros(0),
            err_msg=(
                "Unexpected keypoint score output for prediction with keypoint score "
                f"below threshold got {keypoints_scores_no_pose} instead of {np.zeros(0)}"
            ),
        )
        npt.assert_array_equal(
            x=keypoints_conns_no_pose,
            y=np.zeros(0),
            err_msg=(
                "Unexpected keypoint connection output for prediction with keypoint "
                f"score below threshold got {keypoints_conns_no_pose} instead of {np.zeros(0)}"
            ),
        )
