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
from unittest import mock

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import torch
import yaml

from peekingduck.pipeline.nodes.base import WeightsDownloaderMixin
from peekingduck.pipeline.nodes.model.yolact_edge import Node
from tests.conftest import PKD_DIR, get_groundtruth


GT_RESULTS = get_groundtruth(Path(__file__).resolve())


@pytest.fixture
def yolact_edge_config():
    with open(PKD_DIR / "configs" / "model" / "yolact_edge.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = PKD_DIR
    node_config["score_threshold"] = 0.2
    return node_config


@pytest.fixture(
    params=[
        {"key": "score_threshold", "value": -0.5},
        {"key": "score_threshold", "value": 1.5},
    ],
)
def yolact_edge_bad_config_value(request, yolact_edge_config):
    yolact_edge_config[request.param["key"]] = request.param["value"]
    return yolact_edge_config


@pytest.fixture(params=["r50-fpn", "r101-fpn"])
def yolact_edge_type(request, yolact_edge_config):
    yolact_edge_config["model_type"] = request.param
    return yolact_edge_config


@pytest.mark.mlmodel
class TestYolactEdge:
    def test_no_human_image(self, no_human_image, yolact_edge_type):
        no_human_img = cv2.imread(no_human_image)
        yolact_edge = Node(yolact_edge_type)
        output = yolact_edge.run({"img": no_human_img})
        expected_output = {
            "bboxes": np.empty((0, 4), dtype=np.float32),
            "bbox_labels": np.empty((0)),
            "bbox_scores": np.empty((0), dtype=np.float32),
            "masks": np.empty((0, 0, 0), dtype=np.uint8),
        }
        assert output.keys() == expected_output.keys()
        npt.assert_equal(output["bboxes"], expected_output["bboxes"])
        npt.assert_equal(output["bbox_labels"], expected_output["bbox_labels"])
        npt.assert_equal(output["bbox_scores"], expected_output["bbox_scores"])

    def test_detect_human_bboxes(self, human_image, yolact_edge_config):
        human_img = cv2.imread(human_image)
        yolact_edge = Node(config=yolact_edge_config)
        output = yolact_edge.run({"img": human_img})

        assert "bboxes" in output
        assert output["bboxes"].size > 0

        assert "masks" in output
        assert output["masks"].size > 0

        model_type = yolact_edge.config["model_type"]
        image_name = Path(human_image).stem
        expected = GT_RESULTS[model_type][image_name]

        npt.assert_allclose(output["bboxes"], expected["bboxes"], atol=1e-2)
        npt.assert_equal(output["bbox_labels"], expected["bbox_labels"])
        npt.assert_allclose(output["bbox_scores"], expected["bbox_scores"], atol=1e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
    def test_detect_human_bboxes_gpu(self, human_image, yolact_edge_config):
        human_img = cv2.imread(human_image)
        yolact_edge = Node(config=yolact_edge_config)
        output = yolact_edge.run({"img": human_img})

        assert "bboxes" in output
        assert output["bboxes"].size > 0

        model_type = yolact_edge.config["model_type"]
        image_name = Path(human_image).stem
        expected = GT_RESULTS[model_type][image_name]

        npt.assert_allclose(output["bboxes"], expected["bboxes"], atol=1e-2)
        npt.assert_equal(output["bbox_labels"], expected["bbox_labels"])
        npt.assert_allclose(output["bbox_scores"], expected["bbox_scores"], atol=1e-2)

    def test_get_detect_ids(self, yolact_edge_config):
        yolact_edge = Node(config=yolact_edge_config)
        assert yolact_edge.model.detect_ids == [0]

    def test_invalid_config_detect_ids(self, yolact_edge_config):
        yolact_edge_config["detect"] = 1
        with pytest.raises(TypeError):
            _ = Node(config=yolact_edge_config)

    @mock.patch.object(WeightsDownloaderMixin, "_has_weights", return_value=True)
    def test_invalid_config_model_files(self, _, yolact_edge_config):
        with pytest.raises(ValueError) as excinfo:
            yolact_edge_config["weights"][yolact_edge_config["model_format"]][
                "model_file"
            ][yolact_edge_config["model_type"]] = "some/invalid/path"
            _ = Node(config=yolact_edge_config)
        assert "Model file does not exist. Please check that" in str(excinfo.value)

    def test_invalid_detect_id(self, yolact_edge_config):
        yolact_edge_config["detect"] = 0
        with pytest.raises(TypeError) as excinfo:
            # Passing a non-list detect_id into the config
            _ = Node(config=yolact_edge_config)
        assert "detect_ids has to be a list" == str(excinfo.value)
