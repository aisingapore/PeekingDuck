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

import json
from pathlib import Path
from unittest import TestCase, mock

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import yaml

from peekingduck.pipeline.nodes.base import (
    PEEKINGDUCK_WEIGHTS_SUBDIR,
    WeightsDownloaderMixin,
)
from peekingduck.pipeline.nodes.model.efficientdet import Node
from peekingduck.pipeline.nodes.model.efficientdet_d04.efficientdet_files import (
    detector,
)
from tests.conftest import PKD_DIR

with open(Path(__file__).parent / "test_groundtruth.yml", "r") as infile:
    GT_RESULTS = yaml.safe_load(infile.read())


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


@pytest.fixture
def model_dir(efficientdet_config):
    return (
        efficientdet_config["root"].parent
        / "peekingduck_weights"
        / efficientdet_config["weights"]["model_subdir"]
    )


@pytest.fixture
def class_names(efficientdet_config, model_dir):
    classes_path = model_dir / efficientdet_config["weights"]["classes_file"]
    return {
        val["id"] - 1: val["name"]
        for val in json.loads(Path(classes_path).read_text()).values()
    }


@pytest.fixture(params=[0, 1, 2, 3, 4])
def efficientdet_type(request, efficientdet_config):
    efficientdet_config["model_type"] = request.param
    return efficientdet_config


@pytest.mark.mlmodel
class TestEfficientDet:
    def test_no_human_image(self, test_no_human_images, efficientdet_type):
        blank_image = cv2.imread(test_no_human_images)
        efficientdet = Node(efficientdet_type)
        output = efficientdet.run({"img": blank_image})
        expected_output = {
            "bboxes": np.empty((0, 4), dtype=np.float32),
            "bbox_labels": np.empty((0)),
            "bbox_scores": np.empty((0), dtype=np.float32),
        }
        assert output.keys() == expected_output.keys()
        npt.assert_equal(output["bboxes"], expected_output["bboxes"])
        npt.assert_equal(output["bbox_labels"], expected_output["bbox_labels"])
        npt.assert_equal(output["bbox_scores"], expected_output["bbox_scores"])

    def test_detect_human_bboxes(self, test_human_images, efficientdet_type):
        test_image = cv2.imread(test_human_images)
        efficientdet = Node(efficientdet_type)
        output = efficientdet.run({"img": test_image})

        assert "bboxes" in output
        assert output["bboxes"].size > 0

        model_type = efficientdet.config["model_type"]
        image_name = Path(test_human_images).stem
        expected = GT_RESULTS[model_type][image_name]

        npt.assert_allclose(output["bboxes"], expected["bboxes"], atol=1e-3)
        npt.assert_equal(output["bbox_labels"], expected["bbox_labels"])
        npt.assert_allclose(output["bbox_scores"], expected["bbox_scores"], atol=1e-2)

    def test_efficientdet_preprocess(
        self, create_image, efficientdet_config, model_dir, class_names
    ):
        test_img1 = create_image((720, 1280, 3))
        test_img2 = create_image((640, 480, 3))
        efficientdet_detector = detector.Detector(
            efficientdet_config, model_dir, class_names
        )
        actual_img1, actual_scale1 = efficientdet_detector.preprocess(test_img1, 512)
        actual_img2, actual_scale2 = efficientdet_detector.preprocess(test_img2, 512)

        assert actual_img1.shape == (512, 512, 3)
        assert actual_img2.shape == (512, 512, 3)
        assert actual_img1.dtype == np.float32
        assert actual_img2.dtype == np.float32
        assert actual_scale1 == 0.4
        assert actual_scale2 == 0.8

    def test_efficientdet_postprocess(
        self, efficientdet_config, model_dir, class_names
    ):
        output_bbox = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
        output_label = np.array([0, 0])
        output_score = np.array([0.9, 0.2])
        network_output = (output_bbox, output_score, output_label)
        scale = 0.5
        img_shape = (720, 1280)
        detect_ids = [0]
        efficientdet_detector = detector.Detector(
            efficientdet_config, model_dir, class_names
        )
        boxes, labels, scores = efficientdet_detector.postprocess(
            network_output, scale, img_shape, detect_ids
        )

        expected_bbox = np.array([[1, 2, 3, 4]]) / scale
        expected_bbox[:, [0, 2]] /= img_shape[1]
        expected_bbox[:, [1, 3]] /= img_shape[0]

        expected_score = np.array([0.9])
        npt.assert_almost_equal(expected_bbox, boxes)
        npt.assert_almost_equal(expected_score, scores)
        npt.assert_equal(np.array(["person"]), labels)

    def test_no_weights(self, efficientdet_config, replace_download_weights):
        weights_dir = efficientdet_config["root"].parent / PEEKINGDUCK_WEIGHTS_SUBDIR
        with mock.patch.object(
            WeightsDownloaderMixin, "_has_weights", return_value=False
        ), mock.patch.object(
            WeightsDownloaderMixin, "_download_blob_to", wraps=replace_download_weights
        ), mock.patch.object(
            WeightsDownloaderMixin, "extract_file", wraps=replace_download_weights
        ), TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.yoloxv1.yolox_model.logger"
        ) as captured:
            efficientdet = Node(config=efficientdet_config)
            # records 0 - 20 records are updates to configs
            assert (
                captured.records[0].getMessage()
                == "No weights detected. Proceeding to download..."
            )
            assert (
                captured.records[1].getMessage()
                == f"Weights downloaded to {weights_dir}."
            )
            assert efficientdet is not None

    def test_invalid_config_value(self, efficientdet_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=efficientdet_bad_config_value)
        assert "must be" in str(excinfo.value)
