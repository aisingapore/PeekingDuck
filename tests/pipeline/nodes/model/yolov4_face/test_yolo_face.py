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

from peekingduck.pipeline.nodes.model.yolo_face import Node
from peekingduck.pipeline.nodes.model.yolov4_face.yolo_face_files.detector import (
    Detector,
)


@pytest.fixture
def yolo_config():
    with open(Path(__file__).resolve().parent / "test_yolo_face.yml") as file:
        node_config = yaml.safe_load(file)
    node_config["root"] = Path.cwd()

    return node_config


@pytest.fixture
def model_dir(yolo_config):
    return (
        yolo_config["root"].parent
        / "peekingduck_weights"
        / yolo_config["weights"]["model_subdir"]
    )


@pytest.fixture(params=["v4", "v4tiny"])
def yolo_type(request, yolo_config):
    yolo_config["model_type"] = request.param
    return yolo_config


@pytest.mark.mlmodel
class TestYolo:
    def test_no_human_face_image(self, test_no_human_images, yolo_type):
        blank_image = cv2.imread(test_no_human_images)
        yolo = Node(yolo_type)
        output = yolo.run({"img": blank_image})
        expected_output = {
            "bboxes": np.empty((0, 4), dtype=np.float32),
            "bbox_labels": np.empty((0)),
            "bbox_scores": np.empty((0), dtype=np.float32),
        }
        assert output.keys() == expected_output.keys()
        npt.assert_equal(output["bboxes"], expected_output["bboxes"])
        npt.assert_equal(output["bbox_labels"], expected_output["bbox_labels"])
        npt.assert_equal(output["bbox_scores"], expected_output["bbox_scores"])

    def test_return_at_least_one_face_and_one_bbox(self, test_human_images, yolo_type):
        test_img = cv2.imread(test_human_images)
        test_img = cv2.resize(test_img, (1280, 720))
        yolo = Node(yolo_type)
        output = yolo.run({"img": test_img})
        assert "bboxes" in output
        assert output["bboxes"].size != 0

    def test_no_weights(self, yolo_config, replace_download_weights):
        with mock.patch(
            "peekingduck.weights_utils.checker.has_weights", return_value=False
        ), mock.patch(
            "peekingduck.weights_utils.downloader.download_weights",
            wraps=replace_download_weights,
        ), TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.yolov4_face.yolo_face_model.logger"
        ) as captured:
            yolo = Node(config=yolo_config)
            # records 0 - 20 records are updates to configs
            assert (
                captured.records[0].getMessage()
                == "---no weights detected. proceeding to download...---"
            )
            assert "weights downloaded" in captured.records[1].getMessage()
            assert yolo is not None

    def test_get_detect_ids(self, yolo_type):
        yolo = Node(yolo_type)
        assert yolo.model.get_detect_ids() == [0, 1]

    def test_model_initialization(self, yolo_config, model_dir):
        detector = Detector(yolo_config, model_dir)
        model = detector.yolo
        assert model is not None
