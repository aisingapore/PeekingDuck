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
from unittest import mock, TestCase
from peekingduck.pipeline.nodes.model.movenet import Node
from peekingduck.pipeline.nodes.model.movenetv1 import movenet_model

TEST_DIR = Path.joinpath(Path.cwd(), "images", "testing")
single_model = ["singlepose_lightning", "singlepose_thunder", "multipose_lightning"]
multi_model = ["multipose_lightning"]
single_person_list = ["t2.jpg"]
multi_person_list = ["t1.jpg", "t4.jpg"]
empty_image_list = ["black.jpg", "t3.jpg"]
invalid_thresholds = [-1, 1.1]


@pytest.fixture(params=empty_image_list)
def empty_image(request):
    yield request.param


@pytest.fixture(params=single_person_list)
def single_person_image(request):
    yield request.param


@pytest.fixture(params=multi_person_list)
def multi_person_image(request):
    yield request.param


@pytest.fixture(params=invalid_thresholds)
def invalid_thresholds(request):
    yield request.param


@pytest.fixture
def movenet_config():
    filepath = Path.joinpath(
        Path.cwd(),
        "tests/pipeline/nodes/model/movenetv1/test_movenet.yml",
    )
    with filepath.open() as file:
        node_config = yaml.safe_load(file)
    node_config["root"] = Path.cwd()

    return node_config


@pytest.fixture()
def model(movenet_config):
    model = movenet_model(movenet_config)
    return model


@pytest.fixture(params=single_model)
def movenet_model_single(request, movenet_config):
    movenet_config["model_type"] = request.param
    node = Node(movenet_config)
    return node


@pytest.fixture(params=multi_model)
def movenet_model_multi(request, movenet_config):
    movenet_config["model_type"] = request.param
    node = Node(movenet_config)
    return node


def replace_download_weights(root, blob_file):
    return False


@pytest.mark.mlmodel
class TestMoveNet:
    def test_no_weights(self, movenet_config):
        with mock.patch(
            "peekingduck.weights_utils.checker.has_weights", return_value=False
        ):
            with mock.patch(
                "peekingduck.weights_utils.downloader.download_weights",
                wraps=replace_download_weights,
            ):
                with TestCase.assertLogs(
                    "peekingduck.pipeline.nodes.model.movenet.movenet_model.logger"
                ) as captured:

                    movenet = Node(config=movenet_config)
                    # records 0 - 20 records are updates to configs
                    assert (
                        captured.records[0].getMessage()
                        == "---no weights detected. proceeding to download...---"
                    )
                    assert "weights downloaded" in captured.records[1].getMessage()
                    assert movenet is not None

    def test_no_human_single(self, empty_image, movenet_model_single):
        no_human_img = cv2.imread(str(Path.joinpath(TEST_DIR, empty_image)))
        output = movenet_model_single.run({"img": no_human_img})
        expected_output = {
            "bboxes": np.zeros(0),
            "keypoints": np.zeros(0),
            "keypoint_scores": np.zeros(0),
            "keypoint_conns": np.zeros(0),
            "bbox_labels": np.zeros(0),
        }

        assert output.keys() == expected_output.keys()
        for i in expected_output.keys():
            npt.assert_array_equal(
                output[i], expected_output[i]
            ), "unexpected output for {}".format(i)

    def test_no_human_multi(self, empty_image, movenet_model_multi):
        no_human_img = cv2.imread(str(Path.joinpath(TEST_DIR, empty_image)))
        output = movenet_model_multi.run({"img": no_human_img})
        expected_output = {
            "bboxes": np.zeros(0),
            "keypoints": np.zeros(0),
            "keypoint_scores": np.zeros(0),
            "keypoint_conns": np.zeros(0),
            "bbox_labels": np.zeros(0),
        }

        assert output.keys() == expected_output.keys()
        for i in expected_output.keys():
            npt.assert_array_equal(
                output[i], expected_output[i]
            ), "unexpected output for {}".format(i)

    def test_single_human(self, single_person_image, movenet_model_single):
        single_human_img = cv2.imread(str(Path.joinpath(TEST_DIR, single_person_image)))
        output = movenet_model_single.run({"img": single_human_img})
        expected_output = dict.fromkeys(
            ["bboxes", "keypoints", "keypoint_scores", "keypoint_conns", "bbox_labels"]
        )
        assert output.keys() == expected_output.keys()
        for i in expected_output.keys():
            assert (
                len(output[i]) == 1
            ), "unexpected number of outputs for {} in singlepose".format(i)
        assert output["bbox_labels"] == ["Person"]

    def test_multi_human(self, multi_person_image, movenet_model_multi):
        multi_human_img = cv2.imread(str(Path.joinpath(TEST_DIR, multi_person_image)))
        output = movenet_model_multi.run({"img": multi_human_img})
        expected_output = dict.fromkeys(
            ["bboxes", "keypoints", "keypoint_scores", "keypoint_conns", "bbox_labels"]
        )
        assert output.keys() == expected_output.keys()
        for i in expected_output.keys():
            assert (
                len(output[i]) >= 2
            ), "unexpected number of outputs for {} in multipose".format(i)

        for label in output["bbox_labels"]:
            assert label == "Person"

    def test_bbox_score_threshold(self, movenet_config, invalid_thresholds):
        movenet_config["bbox_score_threshold"] = invalid_thresholds
        with pytest.raises(ValueError):
            movenet = Node(movenet_config)

    def test_keypoint_score_threshold(self, movenet_config, invalid_thresholds):
        movenet_config["keypoint_score_threshold"] = invalid_thresholds
        with pytest.raises(ValueError):
            movenet = Node(movenet_config)

    def test_model_type(self, movenet_config):
        movenet_config["model_type"] = "not_movenet"
        with pytest.raises(ValueError):
            movenet = Node(movenet_config)
