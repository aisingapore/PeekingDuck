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

from peekingduck.pipeline.nodes.model.mtcnn import Node
from peekingduck.pipeline.nodes.model.mtcnnv1.mtcnn_files.detector import Detector

with open(Path(__file__).parent / "test_groundtruth.yml", "r") as infile:
    GT_RESULTS = yaml.safe_load(infile.read())


@pytest.fixture
def mtcnn_config():
    with open(Path(__file__).resolve().parent / "test_mtcnn.yml") as file:
        node_config = yaml.safe_load(file)
    node_config["root"] = Path.cwd()

    return node_config


@pytest.fixture(
    params=[
        {"key": "scale_factor", "value": -0.5},
        {"key": "scale_factor", "value": 1.5},
        {"key": "network_thresholds", "value": [-0.5, -0.5, -0.5]},
        {"key": "network_thresholds", "value": [1.5, 1.5, 1.5]},
        {"key": "score_threshold", "value": -0.5},
        {"key": "score_threshold", "value": 1.5},
    ],
)
def mtcnn_bad_config_value(request, mtcnn_config):
    mtcnn_config[request.param["key"]] = request.param["value"]
    return mtcnn_config


@pytest.fixture(
    params=[{"key": "min_size", "value": -0.5}],
)
def mtcnn_negative_config_value(request, mtcnn_config):
    mtcnn_config[request.param["key"]] = request.param["value"]
    return mtcnn_config


@pytest.fixture
def model_dir(mtcnn_config):
    return (
        mtcnn_config["root"].parent
        / "peekingduck_weights"
        / mtcnn_config["weights"]["model_subdir"]
    )


@pytest.mark.mlmodel
class TestMtcnn:
    def test_no_human_face_image(self, test_no_human_images, mtcnn_config):
        blank_image = cv2.imread(test_no_human_images)
        mtcnn = Node(mtcnn_config)
        output = mtcnn.run({"img": blank_image})
        expected_output = {
            "bboxes": np.empty((0, 4), dtype=np.float32),
            "bbox_scores": np.empty((0), dtype=np.float32),
            "bbox_labels": np.empty((0)),
        }
        assert output.keys() == expected_output.keys()
        npt.assert_equal(output["bboxes"], expected_output["bboxes"])
        npt.assert_equal(output["bbox_scores"], expected_output["bbox_scores"])
        npt.assert_equal(output["bbox_labels"], expected_output["bbox_labels"])

    def test_detect_face_bboxes(self, test_human_images, mtcnn_config):
        test_img = cv2.imread(test_human_images)
        mtcnn = Node(mtcnn_config)
        output = mtcnn.run({"img": test_img})

        assert "bboxes" in output
        assert output["bboxes"].size != 0

        image_name = Path(test_human_images).stem
        expected = GT_RESULTS[image_name]

        npt.assert_allclose(output["bboxes"], expected["bboxes"], atol=1e-3)
        npt.assert_equal(output["bbox_labels"], expected["bbox_labels"])
        npt.assert_allclose(output["bbox_scores"], expected["bbox_scores"], atol=1e-2)

    def test_no_weights(self, mtcnn_config, replace_download_weights):
        with mock.patch(
            "peekingduck.weights_utils.checker.has_weights", return_value=False
        ), mock.patch(
            "peekingduck.weights_utils.downloader.download_weights",
            wraps=replace_download_weights,
        ), TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.mtcnnv1.mtcnn_model.logger"
        ) as captured:
            mtcnn = Node(config=mtcnn_config)
            # records 0 - 20 records are updates to configs
            assert (
                captured.records[0].getMessage()
                == "---no weights detected. proceeding to download...---"
            )
            assert "weights downloaded" in captured.records[1].getMessage()
            assert mtcnn is not None

    def test_invalid_config_value(self, mtcnn_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=mtcnn_bad_config_value)
        assert "must be in [0, 1]" in str(excinfo.value)

    def test_negative_config_value(self, mtcnn_negative_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=mtcnn_negative_config_value)
        assert "must be more than 0" in str(excinfo.value)

    def test_invalid_config_model_files(self, mtcnn_config):
        with mock.patch(
            "peekingduck.weights_utils.checker.has_weights", return_value=True
        ), pytest.raises(ValueError) as excinfo:
            mtcnn_config["weights"]["model_file"] = "some/invalid/path"
            _ = Node(config=mtcnn_config)
        assert "Graph file does not exist. Please check that" in str(excinfo.value)

    def test_model_initialization(self, mtcnn_config, model_dir):
        detector = Detector(mtcnn_config, model_dir)
        model = detector.mtcnn
        assert model is not None
