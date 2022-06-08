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

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import yaml

from peekingduck.pipeline.nodes.model.yolo_license_plate import Node
from tests.conftest import PKD_DIR, get_groundtruth

GT_RESULTS = get_groundtruth(Path(__file__).resolve())


@pytest.fixture
def yolo_config():
    with open(PKD_DIR / "configs" / "model" / "yolo_license_plate.yml") as infile:
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
def yolo_bad_config_value(request, yolo_config):
    yolo_config[request.param["key"]] = request.param["value"]
    return yolo_config


@pytest.fixture(params=["v4", "v4tiny"])
def yolo_type(request, yolo_config):
    yolo_config["model_type"] = request.param
    return yolo_config


@pytest.mark.mlmodel
class TestYOLOLicensePlate:
    def test_no_license_plate_image(self, no_license_plate_image, yolo_type):
        no_license_plate_img = cv2.imread(no_license_plate_image)
        yolo = Node(yolo_type)
        output = yolo.run({"img": no_license_plate_img})
        expected_output = {"bboxes": [], "bbox_labels": [], "bbox_scores": []}
        assert output.keys() == expected_output.keys()
        assert type(output["bboxes"]) == np.ndarray
        assert type(output["bbox_labels"]) == np.ndarray
        assert type(output["bbox_scores"]) == np.ndarray
        assert len(output["bboxes"]) == 0
        assert len(output["bbox_labels"]) == 0
        assert len(output["bbox_scores"]) == 0

    def test_detect_license_plate_image(self, license_plate_image, yolo_type):
        license_plate_img = cv2.imread(license_plate_image)
        yolo = Node(yolo_type)
        output = yolo.run({"img": license_plate_img})

        assert "bboxes" in output
        assert output["bboxes"].size > 0

        model_type = yolo.config["model_type"]
        image_name = Path(license_plate_image).stem
        expected = GT_RESULTS[model_type][image_name]

        npt.assert_allclose(output["bboxes"], expected["bboxes"], atol=1e-3)
        npt.assert_equal(output["bbox_labels"], expected["bbox_labels"])
        npt.assert_allclose(output["bbox_scores"], expected["bbox_scores"], atol=1e-2)

    def test_invalid_config_value(self, yolo_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=yolo_bad_config_value)
        assert "_threshold must be between [0.0, 1.0]" in str(excinfo.value)
