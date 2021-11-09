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
import yaml

from peekingduck.pipeline.nodes.model.yolox import Node


@pytest.fixture
def yolox_config():
    file_path = (
        Path.cwd()
        / "tests"
        / "pipeline"
        / "nodes"
        / "model"
        / "yoloxv1"
        / "test_yolox.yml"
    )
    with open(file_path) as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = Path.cwd()

    return node_config


@pytest.fixture(params=[1, {"some_key": "some_value"}])
def yolox_bad_detect_ids_config(request, yolox_config):
    yolox_config["detect_ids"] = request.param
    return yolox_config


@pytest.fixture(params=[-0.5, 1.5])
def yolox_bad_iou_config(request, yolox_config):
    yolox_config["iou_threshold"] = request.param
    return yolox_config


@pytest.fixture(params=[-0.5, 1.5])
def yolox_bad_score_config(request, yolox_config):
    yolox_config["score_threshold"] = request.param
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


@pytest.fixture
def yolox_default(yolox_config):
    node = Node(yolox_config)

    return node


@pytest.fixture(params=["yolox-l", "yolox-m", "yolox-s", "yolox-tiny"])
def yolox(request, yolox_matrix_config):
    yolox_matrix_config["model_type"] = request.param
    node = Node(yolox_matrix_config)

    return node


def replace_download_weights(root, blob_file):
    pass


@pytest.mark.mlmodel
class TestYOLOX:
    def test_no_human_image(self, test_no_human_images, yolox):
        blank_image = cv2.imread(test_no_human_images)
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

    def test_return_at_least_one_person_and_one_bbox(self, test_human_images, yolox):
        test_image = cv2.imread(test_human_images)
        output = yolox.run({"img": test_image})

        assert "bboxes" in output
        assert output["bboxes"].size > 0

    def test_get_detect_ids(self, yolox_default):
        assert yolox_default.model.detect_ids == [0]

    def test_no_weights(self, yolox_config):
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
                == "No YOLOX weights detected. Proceeding to download..."
            )
            assert (
                captured.records[1].getMessage() == "YOLOX weights download complete."
            )
            assert yolox is not None

    def test_empty_detect_ids(self, yolox_config):
        with TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.yoloxv1.yolox_model.logger"
        ) as captured:
            yolox_config["detect_ids"] = []
            yolox = Node(config=yolox_config)

            assert (
                captured.records[0].getMessage() == "Detecting all available classes."
            )
            assert "IDs being detected: []" in captured.records[1].getMessage()
            assert yolox is not None

    def test_invalid_config_detect_ids(self, yolox_bad_detect_ids_config):
        with pytest.raises(TypeError):
            _ = Node(config=yolox_bad_detect_ids_config)

    def test_invalid_config_iou_threshold(self, yolox_bad_iou_config):
        with pytest.raises(ValueError):
            _ = Node(config=yolox_bad_iou_config)

    def test_invalid_config_score_threshold(self, yolox_bad_score_config):
        with pytest.raises(ValueError):
            _ = Node(config=yolox_bad_score_config)

    def test_invalid_config_model_files(self, yolox_config):
        with mock.patch(
            "peekingduck.weights_utils.checker.has_weights", return_value=True
        ), pytest.raises(ValueError):
            yolox_config["model_files"][
                yolox_config["model_type"]
            ] = "some/invalid/path"
            _ = Node(config=yolox_config)

    def test_invalid_image(self, test_no_human_images, yolox_default):
        blank_image = cv2.imread(test_no_human_images)
        # Potentially passing in a file path or a tuple from image reader
        # output
        with pytest.raises(TypeError):
            _ = yolox_default.run({"img": Path.cwd()})
        with pytest.raises(TypeError):
            _ = yolox_default.run({"img": ("image name", blank_image)})
