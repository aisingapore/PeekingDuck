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

from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import torch
import yaml

from peekingduck.pipeline.nodes.base import WeightsDownloaderMixin
from peekingduck.pipeline.nodes.model.yolox import Node
from tests.conftest import PKD_DIR, get_groundtruth

GT_RESULTS = get_groundtruth(Path(__file__).resolve())


@pytest.fixture
def yolox_config():
    with open(PKD_DIR / "configs" / "model" / "yolox.yml") as infile:
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
        {"agnostic_nms": True, "fuse": True, "half": True},
        {"agnostic_nms": True, "fuse": True, "half": False},
        {"agnostic_nms": True, "fuse": False, "half": True},
        {"agnostic_nms": True, "fuse": False, "half": False},
        {"agnostic_nms": False, "fuse": True, "half": True},
        {"agnostic_nms": False, "fuse": True, "half": False},
        {"agnostic_nms": False, "fuse": False, "half": True},
        {"agnostic_nms": False, "fuse": False, "half": False},
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
    def test_no_human_image(self, no_human_image, yolox_config_cpu):
        no_human_img = cv2.imread(no_human_image)
        yolox = Node(yolox_config_cpu)
        output = yolox.run({"img": no_human_img})
        expected_output = {
            "bboxes": np.empty((0, 4), dtype=np.float32),
            "bbox_labels": np.empty((0)),
            "bbox_scores": np.empty((0), dtype=np.float32),
        }
        assert output.keys() == expected_output.keys()
        npt.assert_equal(output["bboxes"], expected_output["bboxes"])
        npt.assert_equal(output["bbox_labels"], expected_output["bbox_labels"])
        npt.assert_equal(output["bbox_scores"], expected_output["bbox_scores"])

    def test_detect_human_bboxes(self, human_image, yolox_config_cpu):
        human_img = cv2.imread(human_image)
        yolox = Node(yolox_config_cpu)
        output = yolox.run({"img": human_img})

        assert "bboxes" in output
        assert output["bboxes"].size > 0

        model_type = yolox.config["model_type"]
        image_name = Path(human_image).stem
        expected = GT_RESULTS[model_type][image_name]

        npt.assert_allclose(output["bboxes"], expected["bboxes"], atol=1e-3)
        npt.assert_equal(output["bbox_labels"], expected["bbox_labels"])
        npt.assert_allclose(output["bbox_scores"], expected["bbox_scores"], atol=1e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
    def test_detect_human_bboxes_gpu(self, human_image, yolox_matrix_config):
        human_img = cv2.imread(human_image)
        # Ran on YOLOX-tiny only due to GPU OOM error on some systems
        yolox = Node(yolox_matrix_config)
        output = yolox.run({"img": human_img})

        assert "bboxes" in output
        assert output["bboxes"].size > 0

        model_type = yolox.config["model_type"]
        image_name = Path(human_image).stem
        expected = GT_RESULTS[model_type][image_name]

        npt.assert_allclose(output["bboxes"], expected["bboxes"], atol=1e-3)
        npt.assert_equal(output["bbox_labels"], expected["bbox_labels"])
        npt.assert_allclose(output["bbox_scores"], expected["bbox_scores"], atol=1e-2)

    def test_get_detect_ids(self, yolox_config):
        yolox = Node(yolox_config)
        assert yolox.model.detect_ids == [0]

    def test_invalid_config_detect_ids(self, yolox_config):
        yolox_config["detect"] = 1
        with pytest.raises(TypeError):
            _ = Node(config=yolox_config)

    def test_invalid_config_value(self, yolox_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=yolox_bad_config_value)
        assert "_threshold must be between [0.0, 1.0]" in str(excinfo.value)

    @mock.patch.object(WeightsDownloaderMixin, "_has_weights", return_value=True)
    def test_invalid_config_model_files(self, _, yolox_config):
        with pytest.raises(ValueError) as excinfo:
            yolox_config["weights"][yolox_config["model_format"]]["model_file"][
                yolox_config["model_type"]
            ] = "some/invalid/path"
            _ = Node(config=yolox_config)
        assert "Model file does not exist. Please check that" in str(excinfo.value)

    def test_invalid_image(self, no_human_image, yolox_config):
        no_human_img = cv2.imread(no_human_image)
        yolox = Node(yolox_config)
        # Potentially passing in a file path or a tuple from image reader
        # output
        with pytest.raises(TypeError) as excinfo:
            _ = yolox.run({"img": Path.cwd()})
        assert "image must be a np.ndarray" == str(excinfo.value)
        with pytest.raises(TypeError) as excinfo:
            _ = yolox.run({"img": ("image name", no_human_img)})
        assert "image must be a np.ndarray" == str(excinfo.value)
