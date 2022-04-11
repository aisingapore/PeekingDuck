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

TOLERANCE = 1e-6


with open(Path(__file__).resolve().parent / "test_groundtruth.yml") as infile:
    GT_RESULTS = yaml.safe_load(infile)


@pytest.fixture(params=["black.jpg", "t3.jpg"])
def empty_image(request):
    yield str(TEST_IMAGES_DIR / request.param)
    K.clear_session()
    gc.collect()


@pytest.fixture(params=["t2.jpg"])
def single_person_image(request):
    yield str(TEST_IMAGES_DIR / request.param)
    K.clear_session()
    gc.collect()


@pytest.fixture(params=["t1.jpg", "t4.jpg"])
def multi_person_image(request):
    yield str(TEST_IMAGES_DIR / request.param)
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
    return movenet_config


@pytest.fixture(params=["multipose_lightning"])
def movenet_config_multi(request, movenet_config):
    movenet_config["model_type"] = request.param
    return movenet_config


@pytest.mark.mlmodel
class TestMoveNet:
    def test_no_human_single(self, empty_image, movenet_config_single):
        no_human_img = cv2.imread(empty_image)
        model = Node(movenet_config_single)
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
        no_human_img = cv2.imread(empty_image)
        model = Node(movenet_config_multi)
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
        single_human_img = cv2.imread(single_person_image)
        movenet = Node(movenet_config_single)
        output = movenet.run({"img": single_human_img})

        expected_keys = {
            "bboxes",
            "bbox_labels",
            "keypoints",
            "keypoint_conns",
            "keypoint_scores",
        }
        assert set(output.keys()) == expected_keys
        for key in expected_keys:
            assert len(output[key]) == 1, (
                f"unexpected number of detection for {key} in singlepose, "
                f"expected 1 got {len(output[key])}"
            )

        model_type = movenet.config["model_type"]
        image_name = Path(single_person_image).stem
        expected = GT_RESULTS[model_type][image_name]

        npt.assert_allclose(output["bboxes"], expected["bboxes"], atol=TOLERANCE)
        npt.assert_equal(output["bbox_labels"], expected["bbox_labels"])
        npt.assert_allclose(output["keypoints"], expected["keypoints"], atol=TOLERANCE)
        npt.assert_allclose(
            output["keypoint_conns"], expected["keypoint_conns"], atol=TOLERANCE
        )
        npt.assert_allclose(
            output["keypoint_scores"], expected["keypoint_scores"], atol=TOLERANCE
        )

    def test_multi_human(self, multi_person_image, movenet_config_multi):
        multi_human_img = cv2.imread(multi_person_image)
        movenet = Node(movenet_config_multi)
        output = movenet.run({"img": multi_human_img})

        expected_keys = {
            "bboxes",
            "bbox_labels",
            "keypoints",
            "keypoint_conns",
            "keypoint_scores",
        }
        assert set(output.keys()) == expected_keys
        for key in expected_keys:
            assert (
                len(output[key]) > 1
            ), f"unexpected number of detection for {key} in multipose"

        model_type = movenet.config["model_type"]
        image_name = Path(multi_person_image).stem
        expected = GT_RESULTS[model_type][image_name]

        npt.assert_allclose(output["bboxes"], expected["bboxes"], atol=TOLERANCE)
        npt.assert_equal(output["bbox_labels"], expected["bbox_labels"])
        npt.assert_allclose(output["keypoints"], expected["keypoints"], atol=TOLERANCE)

        assert len(output["keypoint_conns"]) == len(expected["keypoint_conns"])
        # Detections can have different number of valid keypoint connections
        # and the keypoint connections result can be a ragged list lists.  When
        # converted to numpy array, the `keypoint_conns`` array will become
        # np.array([list(keypoint connections array), list(next keypoint
        # connections array), ...])
        # Thus, iterate through the detections
        for i, expected_keypoint_conns in enumerate(expected["keypoint_conns"]):
            npt.assert_allclose(
                output["keypoint_conns"][i],
                expected_keypoint_conns,
                atol=TOLERANCE,
            )

        npt.assert_allclose(
            output["keypoint_scores"], expected["keypoint_scores"], atol=TOLERANCE
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
