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
import pytest
import yaml
import cv2
from pathlib import Path
import numpy as np
import numpy.testing as npt
import tensorflow as tf
from peekingduck.pipeline.nodes.model.movenet import Node
from peekingduck.pipeline.nodes.model.movenetv1.movenet_files.predictor import Predictor

TEST_DIR = Path.joinpath(Path.cwd(), "images", "testing")

single_person_list = ["t2.jpg"]


@pytest.fixture
def movenet_config():
    filepath = Path.joinpath(
        Path.cwd(),
        "tests/pipeline/nodes/model/movenetv1/test_movenet.yml",
    )
    with open(filepath) as file:
        node_config = yaml.safe_load(file)
    node_config["root"] = Path.cwd()

    return node_config


@pytest.fixture
def movenet(movenet_config):
    node = Node(movenet_config)
    return node


@pytest.fixture
def model_dir(movenet_config):
    return (
        movenet_config["root"].parent
        / "peekingduck_weights"
        / movenet_config["weights"]["model_subdir"]
    )


@pytest.fixture
def movenet_predictor(movenet_config, model_dir):
    predictor = Predictor(movenet_config, model_dir)
    return predictor


@pytest.fixture(params=single_person_list)
def single_person_image(request):
    yield request.param


@pytest.mark.mlmodel
class TestPredictor:
    def test_predictor(self):
        assert movenet_predictor is not None, "Predictor is not instantiated"

    def test_model_creation(self, movenet_predictor):
        assert (
            movenet_predictor._create_movenet_model is not None
        ), "Model is not loaded"

    def test_get_resolution_as_tuple(self, movenet_predictor):
        resolution = {"height": 256, "width": 256}
        tuple_res = movenet_predictor.get_resolution_as_tuple(resolution)
        assert type(tuple_res) is tuple, "Resolution in config must be a tuple"
        assert tuple_res == (256, 256), "Resolution is loaded incorrectly"
        assert len(tuple_res) == 2, "Resolution in config must be a 2 length tuple"

    def test_predict(self, movenet_predictor, single_person_image):
        img = cv2.imread(str(Path.joinpath(TEST_DIR, single_person_image)))
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

    def test_get_results_single(self, movenet_predictor):
        # prediction for singlepose model is in shape of [1,1,17,13]
        # generates random tensor with values from 0.3 to 0.9
        prediction = tf.random.uniform(
            (1, 1, 17, 3), minval=0.3, maxval=0.9, dtype=tf.dtypes.float32, seed=24
        )
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
        npt.assert_array_equal(bbox_no_pose, np.zeros(0)), "unexpected output for bbox"
        npt.assert_array_equal(
            valid_keypoints_no_pose, np.zeros(0)
        ), "unexpected output for keypoints"
        npt.assert_array_equal(
            keypoints_scores_no_pose, np.zeros(0)
        ), "unexpected output for keypoint scores"
        npt.assert_array_equal(
            keypoints_conns_no_pose, np.zeros(0)
        ), "unexpected output for keypoint connections"

    def test_get_results_multi(self, movenet_predictor):
        # prediction for multi model is in shape of [1,6,56]
        # generates random tensor with values from 0.2 to 0.9
        # since threshold in config is at 0.2, this random tensor
        # will have at least 1 pose after filtering
        prediction = tf.random.uniform(
            (1, 6, 56), minval=0.3, maxval=0.9, dtype=tf.dtypes.float32, seed=24
        )
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
        npt.assert_array_equal(bbox_no_pose, np.zeros(0)), "unexpected output for bbox"
        npt.assert_array_equal(
            valid_keypoints_no_pose, np.zeros(0)
        ), "unexpected output for keypoints"
        npt.assert_array_equal(
            keypoints_scores_no_pose, np.zeros(0)
        ), "unexpected output for keypoint scores"
        npt.assert_array_equal(
            keypoints_conns_no_pose, np.zeros(0)
        ), "unexpected output for keypoint connections"
