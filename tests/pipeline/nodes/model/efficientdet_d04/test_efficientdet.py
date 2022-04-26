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

from peekingduck.pipeline.nodes.model.efficientdet import Node
from tests.conftest import PKD_DIR, get_groundtruth

GT_RESULTS = get_groundtruth(Path(__file__).resolve())


@pytest.fixture
def efficientdet_config():
    with open(PKD_DIR / "configs" / "model" / "efficientdet.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = Path.cwd()

    return node_config


@pytest.fixture(
    params=[
        {"key": "score_threshold", "value": -0.5},
        {"key": "score_threshold", "value": 1.5},
        {"key": "model_type", "value": 5},
        {"key": "model_type", "value": 1.5},
    ],
)
def efficientdet_bad_config_value(request, efficientdet_config):
    efficientdet_config[request.param["key"]] = request.param["value"]
    return efficientdet_config


@pytest.fixture(params=[0, 1, 2, 3, 4])
def efficientdet_type(request, efficientdet_config):
    efficientdet_config["model_type"] = request.param
    return efficientdet_config


@pytest.mark.mlmodel
class TestEfficientDet:
    def test_no_human_image(self, no_human_image, efficientdet_type):
        no_human_img = cv2.imread(no_human_image)
        efficientdet = Node(efficientdet_type)
        output = efficientdet.run({"img": no_human_img})
        expected_output = {
            "bboxes": np.empty((0, 4), dtype=np.float32),
            "bbox_labels": np.empty((0)),
            "bbox_scores": np.empty((0), dtype=np.float32),
        }
        assert output.keys() == expected_output.keys()
        npt.assert_equal(output["bboxes"], expected_output["bboxes"])
        npt.assert_equal(output["bbox_labels"], expected_output["bbox_labels"])
        npt.assert_equal(output["bbox_scores"], expected_output["bbox_scores"])

    def test_detect_human_bboxes(self, human_image, efficientdet_type):
        human_img = cv2.imread(human_image)
        efficientdet = Node(efficientdet_type)
        output = efficientdet.run({"img": human_img})

        assert "bboxes" in output
        assert output["bboxes"].size > 0

        model_type = efficientdet.config["model_type"]
        image_name = Path(human_image).stem
        expected = GT_RESULTS[model_type][image_name]

        npt.assert_allclose(output["bboxes"], expected["bboxes"], atol=1e-3)
        npt.assert_equal(output["bbox_labels"], expected["bbox_labels"])
        npt.assert_allclose(output["bbox_scores"], expected["bbox_scores"], atol=1e-2)

    def test_efficientdet_preprocess(self, create_image, efficientdet_config):
        test_img1 = create_image((720, 1280, 3))
        test_img2 = create_image((640, 480, 3))
        efficientdet = Node(efficientdet_config)

        actual_img1, actual_scale1 = efficientdet.model.detector._preprocess(test_img1)
        actual_img2, actual_scale2 = efficientdet.model.detector._preprocess(test_img2)

        assert actual_img1.shape == (512, 512, 3)
        assert actual_img2.shape == (512, 512, 3)
        assert actual_img1.dtype == np.float32
        assert actual_img2.dtype == np.float32
        assert actual_scale1 == 0.4
        assert actual_scale2 == 0.8

    def test_efficientdet_postprocess(self, efficientdet_config):
        output_bbox = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
        output_label = np.array([0, 0])
        output_score = np.array([0.9, 0.2])
        network_output = (output_bbox, output_score, output_label)
        scale = 0.5
        img_shape = (720, 1280)
        efficientdet = Node(efficientdet_config)

        boxes, labels, scores = efficientdet.model.detector._postprocess(
            network_output, scale, img_shape
        )

        expected_bbox = np.array([[1, 2, 3, 4]]) / scale
        expected_bbox[:, [0, 2]] /= img_shape[1]
        expected_bbox[:, [1, 3]] /= img_shape[0]

        expected_score = np.array([0.9])
        npt.assert_almost_equal(expected_bbox, boxes)
        npt.assert_almost_equal(expected_score, scores)
        npt.assert_equal(np.array(["person"]), labels)

    def test_invalid_config_value(self, efficientdet_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=efficientdet_bad_config_value)
        assert "must be" in str(excinfo.value)
