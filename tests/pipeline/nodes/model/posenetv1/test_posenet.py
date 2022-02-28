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

from pathlib import Path
from unittest import TestCase, mock

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import yaml

from peekingduck.pipeline.nodes.model.posenet import Node


@pytest.fixture
def posenet_config():
    filepath = Path(__file__).resolve().parent / "test_posenet.yml"
    with open(filepath) as file:
        node_config = yaml.safe_load(file)
    node_config["root"] = Path.cwd()
    return node_config


@pytest.fixture(params=[50, 75, 100, "resnet"])
def posenet_type(request, posenet_config):
    posenet_config["model_type"] = request.param
    return posenet_config


@pytest.fixture(
    params=[
        {"key": "score_threshold", "value": -0.5},
        {"key": "score_threshold", "value": 1.5},
        {"key": "model_type", "value": 101},
        {"key": "model_type", "value": "inception"},
    ],
)
def posenet_bad_config_value(request, posenet_config):
    posenet_config[request.param["key"]] = request.param["value"]
    return posenet_config


@pytest.mark.mlmodel
class TestPoseNet:
    def test_no_detection(self, test_no_human_images, posenet_type):
        blank_image = cv2.imread(str(test_no_human_images))
        posenet = Node(posenet_type)
        output = posenet.run({"img": blank_image})
        expected_output = {
            "bboxes": np.zeros(0),
            "keypoints": np.zeros(0),
            "keypoint_scores": np.zeros(0),
            "keypoint_conns": np.zeros(0),
            "bbox_labels": np.zeros(0),
        }
        assert output.keys() == expected_output.keys(), "missing keys"
        for i in expected_output.keys():
            npt.assert_array_equal(
                output[i], expected_output[i]
            ), "unexpected output for {}".format(i)

    def test_different_models(self, test_human_images, posenet_type):
        pose_image = cv2.imread(str(test_human_images))
        posenet = Node(posenet_type)
        output = posenet.run({"img": pose_image})
        expected_output = dict.fromkeys(
            ["bboxes", "keypoints", "keypoint_scores", "keypoint_conns", "bbox_labels"]
        )
        assert output.keys() == expected_output.keys(), "missing keys"
        for i in expected_output.keys():
            assert len(output[i]) >= 1, "unexpected number of outputs for {}".format(i)
        for label in output["bbox_labels"]:
            assert label == "Person"

    def test_no_weights(self, posenet_config, replace_download_weights):
        with mock.patch(
            "peekingduck.weights_utils.checker.has_weights", return_value=False
        ), mock.patch(
            "peekingduck.weights_utils.downloader.download_weights",
            wraps=replace_download_weights,
        ), TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.posenetv1.posenet_model.logger"
        ) as captured:
            posenet = Node(config=posenet_config)
            # records 0 - 20 records are updates to configs
            assert (
                captured.records[0].getMessage()
                == "---no weights detected. proceeding to download...---"
            )
            assert "weights downloaded" in captured.records[1].getMessage()
            assert posenet is not None

    def test_invalid_config_value(self, posenet_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=posenet_bad_config_value)
        assert "must be" in str(excinfo.value)
