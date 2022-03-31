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
import pytest
import yaml

from peekingduck.pipeline.nodes.base import (
    PEEKINGDUCK_WEIGHTS_SUBDIR,
    WeightsDownloaderMixin,
)
from peekingduck.pipeline.nodes.model.yolo_license_plate import Node


@pytest.fixture
def yolo_config():
    with open(
        Path(__file__).resolve().parent / "test_yolov4_license_plate.yml"
    ) as file:
        node_config = yaml.safe_load(file)
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
def yolo_bad_config_value(request, yolo_config):
    yolo_config[request.param["key"]] = request.param["value"]
    return yolo_config


@pytest.fixture(params=["v4", "v4tiny"])
def yolo_type(request, yolo_config):
    yolo_config["model_type"] = request.param
    return yolo_config


@pytest.mark.mlmodel
class TestYOLOLicensePlate:
    def test_no_lp_image(self, test_no_lp_images, yolo_type):
        blank_image = cv2.imread(test_no_lp_images)
        yolo = Node(yolo_type)
        output = yolo.run({"img": blank_image})
        expected_output = {"bboxes": [], "bbox_labels": [], "bbox_scores": []}
        assert output.keys() == expected_output.keys()
        assert type(output["bboxes"]) == np.ndarray
        assert type(output["bbox_labels"]) == np.ndarray
        assert type(output["bbox_scores"]) == np.ndarray
        assert len(output["bboxes"]) == 0
        assert len(output["bbox_labels"]) == 0
        assert len(output["bbox_scores"]) == 0

    def test_at_least_one_lp_image(self, test_lp_images, yolo_type):
        test_img = cv2.imread(test_lp_images)
        yolo = Node(yolo_type)
        output = yolo.run({"img": test_img})
        assert "bboxes" in output
        assert len(output["bboxes"]) != 0
        assert len(output["bboxes"]) == len(output["bbox_labels"])

    def test_no_weights(self, yolo_config, replace_download_weights):
        weights_dir = yolo_config["root"].parent / PEEKINGDUCK_WEIGHTS_SUBDIR
        with mock.patch.object(
            WeightsDownloaderMixin, "_has_weights", return_value=False
        ), mock.patch.object(
            WeightsDownloaderMixin, "_download_blob_to", wraps=replace_download_weights
        ), mock.patch.object(
            WeightsDownloaderMixin, "extract_file", wraps=replace_download_weights
        ), TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.yolov4_license_plate.yolo_license_plate_model.logger"
        ) as captured:
            yolo = Node(config=yolo_config)
            # records 0 - 20 records are updates to configs
            assert (
                captured.records[0].getMessage()
                == "No weights detected. Proceeding to download..."
            )
            assert (
                captured.records[1].getMessage()
                == f"Weights downloaded to {weights_dir}."
            )
            assert yolo is not None

    def test_invalid_config_value(self, yolo_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=yolo_bad_config_value)
        assert "_threshold must be between [0, 1]" in str(excinfo.value)
