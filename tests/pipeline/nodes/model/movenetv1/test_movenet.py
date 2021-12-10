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
import gc
import pytest
import yaml
import cv2
from pathlib import Path
import numpy as np
import numpy.testing as npt
from unittest import mock, TestCase
from peekingduck.pipeline.nodes.model.movenet import Node
from peekingduck.pipeline.nodes.model.movenetv1 import movenet_model
import tensorflow.keras.backend as K

TEST_DIR = Path.joinpath(Path.cwd(), "images", "testing")
TOLERANCE = 1e-2
singlepose_models = [
    "singlepose_lightning",
    "singlepose_thunder",
    "multipose_lightning",
]
multipose_models = ["multipose_lightning"]
single_person_images = ["t2.jpg"]
multi_persons_images = ["t1.jpg", "t4.jpg"]
zero_persons_images = ["black.jpg", "t3.jpg"]
invalid_score_thresholds = [-1, 1.1]


@pytest.fixture(params=zero_persons_images)
def empty_image(request):
    yield request.param
    K.clear_session()
    gc.collect()


@pytest.fixture(params=single_person_images)
def single_person_image(request):
    yield request.param
    K.clear_session()
    gc.collect()


@pytest.fixture(params=multi_persons_images)
def multi_person_image(request):
    yield request.param
    K.clear_session()
    gc.collect()


@pytest.fixture(params=invalid_score_thresholds)
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


@pytest.fixture
def ground_truth():
    filepath = Path.joinpath(
        Path.cwd(),
        "tests/pipeline/nodes/model/movenetv1/test_groundtruth.yml",
    )
    with filepath.open() as file:
        ground_truth = yaml.safe_load(file)
    return ground_truth


@pytest.fixture()
def model(movenet_config):
    model = movenet_model(movenet_config)
    return model


@pytest.fixture(params=singlepose_models)
def movenet_model_single(request, movenet_config):
    movenet_config["model_type"] = request.param
    node = Node(movenet_config)
    return node, movenet_config["model_type"]


@pytest.fixture(params=multipose_models)
def movenet_model_multi(request, movenet_config):
    movenet_config["model_type"] = request.param
    node = Node(movenet_config)
    return node, movenet_config["model_type"]


def replace_download_weights(root, blob_file):
    return False


@pytest.mark.mlmodel
class TestMoveNet:
    def test_no_human_single(self, empty_image, movenet_model_single):
        no_human_img = cv2.imread(str(Path.joinpath(TEST_DIR, empty_image)))
        model, _ = movenet_model_single
        output = model.run({"img": no_human_img})
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
                x=output[i],
                y=expected_output[i],
                err_msg=(
                    f"unexpected output for {i}, Expected {np.zeros(0)} \
                        got {output[i]}"
                ),
            )

    def test_no_weights(self, movenet_config):
        with mock.patch(
            "peekingduck.weights_utils.checker.has_weights", return_value=False
        ):
            with mock.patch(
                "peekingduck.weights_utils.downloader.download_weights",
                wraps=replace_download_weights,
            ):
                with TestCase.assertLogs(
                    "peekingduck.pipeline.nodes.model.movenetv1.movenet_model.logger"
                ) as captured:
                    movenet = Node(config=movenet_config)
                    # records 0 - 20 records are updates to configs
                    assert (
                        captured.records[0].getMessage()
                        == "---no weights detected. proceeding to download...---"
                    )
                    assert "weights downloaded" in captured.records[1].getMessage()
                    assert movenet is not None

    def test_no_human_multi(self, empty_image, movenet_model_multi):
        no_human_img = cv2.imread(str(Path.joinpath(TEST_DIR, empty_image)))
        model, _ = movenet_model_multi
        output = model.run({"img": no_human_img})
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
                x=output[i],
                y=expected_output[i],
                err_msg=(
                    f"unexpected output for {i}, Expected {np.zeros(0)} \
                        got {output[i]}"
                ),
            )

    def test_single_human(
        self, single_person_image, movenet_model_single, ground_truth
    ):
        single_human_img = cv2.imread(str(Path.joinpath(TEST_DIR, single_person_image)))
        model, model_type = movenet_model_single
        output = model.run({"img": single_human_img})
        expected_output = dict.fromkeys(
            ["bboxes", "keypoints", "keypoint_scores", "keypoint_conns", "bbox_labels"]
        )
        assert output.keys() == expected_output.keys()
        for i in expected_output.keys():
            assert (
                len(output[i]) == 1
            ), f"unexpected number of detection for {i} in singlepose, expected 1 got {len(output[i])}"
        assert output["bbox_labels"] == ["Person"]
        ground_truth_bboxes = np.asarray(
            ground_truth[model_type][single_person_image]["bboxes"]
        )
        ground_truth_keypoints = np.asarray(
            ground_truth[model_type][single_person_image]["keypoints"]
        )
        ground_truth_keypoint_scores = np.asarray(
            ground_truth[model_type][single_person_image]["keypoint_scores"]
        )
        ground_truth_keypoint_conns = np.asarray(
            ground_truth[model_type][single_person_image]["keypoint_conns"]
        )
        ground_truth_bbox_labels = np.asarray(
            ground_truth[model_type][single_person_image]["bbox_labels"]
        )
        npt.assert_allclose(
            actual=output["bboxes"],
            desired=ground_truth_bboxes,
            atol=TOLERANCE,
            err_msg=(
                f"output for bboxes exceed tolerance of {TOLERANCE} from ground truth: \
                expected {ground_truth_bboxes} got {output['bboxes']}"
            ),
        ),
        npt.assert_allclose(
            actual=output["keypoints"],
            desired=ground_truth_keypoints,
            atol=TOLERANCE,
            err_msg=(
                f"output for keypoints exceed tolerance of {TOLERANCE} from ground truth: \
                expected {ground_truth_keypoints} got {output['keypoints']}"
            ),
        ),
        npt.assert_allclose(
            actual=output["keypoint_scores"],
            desired=ground_truth_keypoint_scores,
            atol=TOLERANCE,
            err_msg=(
                f"output for keypoint_scores exceed tolerance of {TOLERANCE} from ground truth: \
                expected {ground_truth_keypoint_scores} got {output['keypoint_scores']}"
            ),
        ),
        npt.assert_allclose(
            actual=output["keypoint_conns"],
            desired=ground_truth_keypoint_conns,
            atol=TOLERANCE,
            err_msg=(
                f"output for keypoint_conns exceed tolerance of {TOLERANCE} from ground truth: \
                expected {ground_truth_keypoint_conns} got {output['keypoint_conns']}"
            ),
        ),
        assert (
            output["bbox_labels"] == ground_truth_bbox_labels
        ).all(), f"unexpected output for bbox_labels: expected {ground_truth_bbox_labels} got {output['bbox_labels']}"

    def test_multi_human(self, multi_person_image, movenet_model_multi, ground_truth):
        multi_human_img = cv2.imread(str(Path.joinpath(TEST_DIR, multi_person_image)))
        model, model_type = movenet_model_multi
        output = model.run({"img": multi_human_img})
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
        ground_truth_bboxes = np.asarray(
            ground_truth[model_type][multi_person_image]["bboxes"]
        )
        ground_truth_keypoints = np.asarray(
            ground_truth[model_type][multi_person_image]["keypoints"]
        )
        ground_truth_keypoint_scores = np.asarray(
            ground_truth[model_type][multi_person_image]["keypoint_scores"]
        )
        ground_truth_keypoint_conns = np.asarray(
            ground_truth[model_type][multi_person_image]["keypoint_conns"]
        )
        ground_truth_bbox_labels = np.asarray(
            ground_truth[model_type][multi_person_image]["bbox_labels"]
        )
        npt.assert_allclose(
            actual=output["bboxes"],
            desired=ground_truth_bboxes,
            atol=TOLERANCE,
            rtol=TOLERANCE,
            err_msg=(
                f"output for bboxes exceed tolerance of {TOLERANCE} from ground truth: \
                expected {ground_truth_bboxes} got {output['bboxes']}"
            ),
        ),
        npt.assert_allclose(
            actual=output["keypoints"],
            desired=ground_truth_keypoints,
            atol=TOLERANCE,
            rtol=TOLERANCE,
            err_msg=(
                f"output for keypoints exceed tolerance of {TOLERANCE} from ground truth: \
                expected {ground_truth_keypoints} got {output['keypoints']}"
            ),
        ),
        npt.assert_allclose(
            actual=output["keypoint_scores"],
            desired=ground_truth_keypoint_scores,
            atol=TOLERANCE,
            rtol=TOLERANCE,
            err_msg=(
                f"output for keypoint_scores exceed tolerance of {TOLERANCE} from ground truth: \
                expected {ground_truth_keypoint_scores} got {output['keypoint_scores']}"
            ),
        ),
        # Due to the different detections plausibly having difference number of valid keypoints conns.
        # The array of keypoint_conns will come in form of
        # np.array([List(array of keypoints conns),List(array of next keypoints conns),..]) when
        # a list of different length of lists are converted to numpy array.
        # Thus, iterate through the detections and convert the inner List into numpy array before
        # assert testing.
        for i in range(output["keypoint_conns"].shape[0]):
            npt.assert_allclose(
                actual=np.asarray(output["keypoint_conns"][i]),
                desired=np.asarray(ground_truth_keypoint_conns[i]),
                atol=TOLERANCE,
                rtol=TOLERANCE,
                err_msg=(
                    f"output for {i}th idx detection, keypoint_conns exceed tolerance of {TOLERANCE} from ground truth: \
                    expected {np.asarray(ground_truth_keypoint_conns[i])} got {np.asarray(output['keypoint_conns'][i])}"
                ),
            ),
        assert (
            output["bbox_labels"] == ground_truth_bbox_labels
        ).all(), f"unexpected output for bbox_labels: expected {ground_truth_bbox_labels} got {output['bbox_labels']}"

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
