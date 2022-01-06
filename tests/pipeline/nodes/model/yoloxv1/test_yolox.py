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

from pathlib import Path
from unittest import TestCase, mock

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import torch
import yaml

from peekingduck.pipeline.nodes.model.yolox import Node
from peekingduck.weights_utils.finder import PEEKINGDUCK_WEIGHTS_SUBDIR

with open(Path(__file__).parent / "test_groundtruth.yml", "r") as infile:
    GT_RESULTS = yaml.safe_load(infile.read())


@pytest.fixture
def yolox_config():
    file_path = Path(__file__).resolve().parent / "test_yolox.yml"
    with open(file_path) as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = Path.cwd()

    return node_config


@pytest.fixture(
    params=[
        {"key": "iou_threshold", "value": -0.5},
        {"key": "iou_threshold", "value": 1.5},
        {"key": "score_threshold", "value": -0.5},
        {"key": "score_threshold", "value": 1.5},
    ],
)
def yolox_bad_config_value(request, yolox_config):
    yolox_config[request.param["key"]] = request.param["value"]
    return yolox_config


@pytest.fixture(
    params=[
        {"fuse": True, "half": True},
        {"fuse": True, "half": False},
        {"fuse": False, "half": True},
        {"fuse": False, "half": False},
    ]
)
def yolox_matrix_config(request, yolox_config):
    yolox_config.update(request.param)
    return yolox_config


@pytest.fixture(params=["yolox-l", "yolox-m", "yolox-s", "yolox-tiny"])
def yolox_config_cpu(request, yolox_matrix_config):
    yolox_matrix_config["model_type"] = request.param
    with mock.patch("torch.cuda.is_available", return_value=False):
        yield yolox_matrix_config


@pytest.mark.mlmodel
class TestYOLOX:
    def test_no_human_image(self, test_no_human_images, yolox_config_cpu):
        blank_image = cv2.imread(test_no_human_images)
        yolox = Node(yolox_config_cpu)
        output = yolox.run({"img": blank_image})
        expected_output = {
            "bboxes": np.empty((0, 4), dtype=np.float32),
            "bbox_labels": np.empty((0)),
            "bbox_scores": np.empty((0), dtype=np.float32),
        }
        assert output.keys() == expected_output.keys()
        npt.assert_equal(output["bboxes"], expected_output["bboxes"])
        npt.assert_equal(output["bbox_labels"], expected_output["bbox_labels"])
        npt.assert_equal(output["bbox_scores"], expected_output["bbox_scores"])

    def test_detect_human_bboxes(self, test_human_images, yolox_config_cpu):
        test_image = cv2.imread(test_human_images)
        yolox = Node(yolox_config_cpu)
        output = yolox.run({"img": test_image})

        assert "bboxes" in output
        assert output["bboxes"].size > 0

        model_type = yolox.config["model_type"]
        image_name = Path(test_human_images).stem
        expected = GT_RESULTS[model_type][image_name]

        npt.assert_allclose(output["bboxes"], expected["bboxes"], atol=1e-3)
        npt.assert_equal(output["bbox_labels"], expected["bbox_labels"])
        npt.assert_allclose(output["bbox_scores"], expected["bbox_scores"], atol=1e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
    def test_detect_human_bboxes_gpu(self, test_human_images, yolox_matrix_config):
        test_image = cv2.imread(test_human_images)
        # Ran on YOLOX-tiny only due to GPU OOM error on some systems
        yolox = Node(yolox_matrix_config)
        output = yolox.run({"img": test_image})

        assert "bboxes" in output
        assert output["bboxes"].size > 0

        model_type = yolox.config["model_type"]
        image_name = Path(test_human_images).stem
        expected = GT_RESULTS[model_type][image_name]

        npt.assert_allclose(output["bboxes"], expected["bboxes"], atol=1e-3)
        npt.assert_equal(output["bbox_labels"], expected["bbox_labels"])
        npt.assert_allclose(output["bbox_scores"], expected["bbox_scores"], atol=1e-2)

    def test_get_detect_ids(self, yolox_config):
        yolox = Node(yolox_config)
        assert yolox.model.detect_ids == [0]

    def test_no_weights(self, yolox_config, replace_download_weights):
        weights_dir = yolox_config["root"].parent / PEEKINGDUCK_WEIGHTS_SUBDIR
        with mock.patch(
            "peekingduck.weights_utils.checker.has_weights", return_value=False
        ), mock.patch(
            "peekingduck.weights_utils.downloader.download_weights",
            wraps=replace_download_weights,
        ), TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.yoloxv1.yolox_model.logger"
        ) as captured:
            yolox = Node(config=yolox_config)
            # records 0 - 20 records are updates to configs
            assert (
                captured.records[0].getMessage()
                == "No weights detected. Proceeding to download..."
            )
            assert (
                captured.records[1].getMessage()
                == f"Weights downloaded to {weights_dir}."
            )
            assert yolox is not None

    def test_empty_detect_ids(self, yolox_config):
        with TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.yoloxv1.yolox_model.logger"
        ) as captured:
            yolox_config["detect_ids"] = []
            yolox = Node(config=yolox_config)

            assert "IDs being detected: []" in captured.records[0].getMessage()
            assert captured.records[1].getMessage() == "Detecting all YOLOX classes"
            assert yolox is not None

    @pytest.mark.parametrize("detect_ids", [1, {"some_key": "some_value"}])
    def test_invalid_config_detect_ids(self, yolox_config, detect_ids):
        yolox_config["detect_ids"] = detect_ids
        with pytest.raises(TypeError):
            _ = Node(config=yolox_config)

    def test_invalid_config_value(self, yolox_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=yolox_bad_config_value)
        assert "_threshold must be in [0, 1]" in str(excinfo.value)

    def test_invalid_config_model_files(self, yolox_config):
        with mock.patch(
            "peekingduck.weights_utils.checker.has_weights", return_value=True
        ), pytest.raises(ValueError) as excinfo:
            yolox_config["weights"]["model_file"][
                yolox_config["model_type"]
            ] = "some/invalid/path"
            _ = Node(config=yolox_config)
        assert "Model file does not exist. Please check that" in str(excinfo.value)

    def test_invalid_image(self, test_no_human_images, yolox_config):
        blank_image = cv2.imread(test_no_human_images)
        yolox = Node(yolox_config)
        # Potentially passing in a file path or a tuple from image reader
        # output
        with pytest.raises(TypeError) as excinfo:
            _ = yolox.run({"img": Path.cwd()})
        assert "image must be a np.ndarray" == str(excinfo.value)
        with pytest.raises(TypeError) as excinfo:
            _ = yolox.run({"img": ("image name", blank_image)})
        assert "image must be a np.ndarray" == str(excinfo.value)
