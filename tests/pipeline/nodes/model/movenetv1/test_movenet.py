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

import gc
from pathlib import Path
from unittest import TestCase, mock

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import tensorflow.keras.backend as K
import yaml

from peekingduck.pipeline.nodes.base import (
    PEEKINGDUCK_WEIGHTS_SUBDIR,
    WeightsDownloaderMixin,
)
from peekingduck.pipeline.nodes.model.movenet import Node
from tests.conftest import PKD_DIR, TEST_IMAGES_DIR, do_nothing

TEST_IMAGES_DIR = Path.cwd() / "tests" / "data" / "images"
TOLERANCE = 1e-2


with open(Path(__file__).resolve().parent / "test_groundtruth.yml") as infile:
    GT_RESULTS = yaml.safe_load(infile)


@pytest.fixture(params=["black.jpg", "t3.jpg"])
def empty_image(request):
    yield request.param
    K.clear_session()
    gc.collect()


@pytest.fixture(params=["t2.jpg"])
def single_person_image(request):
    yield request.param
    K.clear_session()
    gc.collect()


@pytest.fixture(params=["t1.jpg", "t4.jpg"])
def multi_person_image(request):
    yield request.param
    K.clear_session()
    gc.collect()


@pytest.fixture
def movenet_config():
    with open(PKD_DIR / "configs" / "model" / "movenet.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = Path.cwd()

    return node_config


@pytest.fixture(
    params=[
        {"key": "bbox_score_threshold", "value": -0.5},
        {"key": "bbox_score_threshold", "value": 1.5},
        {"key": "keypoint_score_threshold", "value": -0.5},
        {"key": "keypoint_score_threshold", "value": 1.5},
        {"key": "model_type", "value": "bad_model_type"},
    ],
)
def movenet_bad_config_value(request, movenet_config):
    movenet_config[request.param["key"]] = request.param["value"]
    return movenet_config


@pytest.fixture(
    params=["singlepose_lightning", "singlepose_thunder", "multipose_lightning"]
)
def movenet_config_single(request, movenet_config):
    movenet_config["model_type"] = request.param
    return movenet_config, movenet_config["model_type"]


@pytest.fixture(params=["multipose_lightning"])
def movenet_config_multi(request, movenet_config):
    movenet_config["model_type"] = request.param
    return movenet_config, movenet_config["model_type"]


@pytest.mark.mlmodel
class TestMoveNet:
    def test_no_human_single(self, empty_image, movenet_config_single):
        no_human_img = cv2.imread(str(TEST_IMAGES_DIR / empty_image))
        movenet_config, _ = movenet_config_single
        model = Node(movenet_config)
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
                    f"unexpected output for {i}, Expected {np.zeros(0)} got {output[i]}"
                ),
            )

    def test_no_human_multi(self, empty_image, movenet_config_multi):
        no_human_img = cv2.imread(str(TEST_IMAGES_DIR / empty_image))
        movenet_config, _ = movenet_config_multi
        model = Node(movenet_config)
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
                    f"unexpected output for {i}, Expected {np.zeros(0)} got {output[i]}"
                ),
            )

    def test_single_human(self, single_person_image, movenet_config_single):
        single_human_img = cv2.imread(str(TEST_IMAGES_DIR / single_person_image))
        movenet_config, model_type = movenet_config_single
        model = Node(movenet_config)
        output = model.run({"img": single_human_img})
        expected_output = dict.fromkeys(
            ["bboxes", "keypoints", "keypoint_scores", "keypoint_conns", "bbox_labels"]
        )
        assert output.keys() == expected_output.keys()
        for i in expected_output.keys():
            assert len(output[i]) == 1, (
                f"unexpected number of detection for {i} in singlepose, "
                f"expected 1 got {len(output[i])}"
            )
        assert output["bbox_labels"] == ["person"]
        ground_truth = GT_RESULTS[model_type][single_person_image]
        for key in ("bboxes", "keypoints", "keypoint_scores", "keypoint_conns"):
            gt_values = np.asarray(ground_truth[key])
            npt.assert_allclose(
                actual=output[key],
                desired=gt_values,
                atol=TOLERANCE,
                err_msg=(
                    f"output for {key} exceed tolerance of {TOLERANCE} from ground truth: "
                    f"expected {gt_values} got {output[key]}"
                ),
            )
        ground_truth_bbox_labels = np.asarray(ground_truth["bbox_labels"])
        assert (output["bbox_labels"] == ground_truth_bbox_labels).all(), (
            f"unexpected output for bbox_labels: expected {ground_truth_bbox_labels} "
            f"got {output['bbox_labels']}"
        )

    def test_multi_human(self, multi_person_image, movenet_config_multi):
        multi_human_img = cv2.imread(str(TEST_IMAGES_DIR / multi_person_image))
        movenet_config, model_type = movenet_config_multi
        model = Node(movenet_config)
        output = model.run({"img": multi_human_img})
        expected_output = dict.fromkeys(
            ["bboxes", "keypoints", "keypoint_scores", "keypoint_conns", "bbox_labels"]
        )
        assert output.keys() == expected_output.keys()
        for i in expected_output.keys():
            assert (
                len(output[i]) >= 2
            ), f"unexpected number of outputs for {i} in multipose"
        npt.assert_array_equal(output["bbox_labels"], "person")
        ground_truth = GT_RESULTS[model_type][multi_person_image]
        for key in ("bboxes", "keypoints", "keypoint_scores"):
            gt_values = ground_truth[key]
            npt.assert_allclose(
                actual=output[key],
                desired=gt_values,
                atol=TOLERANCE,
                rtol=TOLERANCE,
                err_msg=(
                    f"output for {key} exceed tolerance of {TOLERANCE} from ground truth: "
                    f"expected {gt_values} got {output[key]}"
                ),
            )

        # Due to the different detections plausibly having difference number of
        # valid keypoints conns.
        # The array of keypoint_conns will come in form of
        # np.array([List(array of keypoints conns),
        # List(array of next keypoints conns),..]) when a list of different
        # length of lists are converted to numpy array.
        # Thus, iterate through the detections and convert the inner List into
        # numpy array before assert testing.
        ground_truth_keypoint_conns = np.asarray(ground_truth["keypoint_conns"])
        for i in range(output["keypoint_conns"].shape[0]):
            npt.assert_allclose(
                actual=np.asarray(output["keypoint_conns"][i]),
                desired=np.asarray(ground_truth_keypoint_conns[i]),
                atol=TOLERANCE,
                rtol=TOLERANCE,
                err_msg=(
                    f"output for {i}th idx detection, keypoint_conns exceed tolerance of {TOLERANCE} from ground truth: "
                    f"expected {np.asarray(ground_truth_keypoint_conns[i])} got {np.asarray(output['keypoint_conns'][i])}"
                ),
            )
        ground_truth_bbox_labels = np.asarray(ground_truth["bbox_labels"])
        assert (output["bbox_labels"] == ground_truth_bbox_labels).all(), (
            f"unexpected output for bbox_labels: expected {ground_truth_bbox_labels} "
            f"got {output['bbox_labels']}"
        )

    @mock.patch.object(WeightsDownloaderMixin, "_has_weights", return_value=False)
    @mock.patch.object(WeightsDownloaderMixin, "_download_blob_to", wraps=do_nothing)
    @mock.patch.object(WeightsDownloaderMixin, "extract_file", wraps=do_nothing)
    def test_no_weights(
        self,
        _,
        mock_download_blob_to,
        mock_extract_file,
        movenet_config,
    ):
        weights_dir = movenet_config["root"].parent / PEEKINGDUCK_WEIGHTS_SUBDIR
        with TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.movenetv1.movenet_model.logger"
        ) as captured:
            movenet = Node(config=movenet_config)
            # records 0 - 20 records are updates to configs
            assert (
                captured.records[0].getMessage()
                == "No weights detected. Proceeding to download..."
            )
            assert (
                captured.records[1].getMessage()
                == f"Weights downloaded to {weights_dir}."
            )
            assert movenet is not None

        assert mock_download_blob_to.called
        assert mock_extract_file.called

    def test_invalid_config_value(self, movenet_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=movenet_bad_config_value)
        assert "must be" in str(excinfo.value)
