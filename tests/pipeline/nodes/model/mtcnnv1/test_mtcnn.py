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
import yaml

from peekingduck.pipeline.nodes.base import WeightsDownloaderMixin
from peekingduck.pipeline.nodes.model.mtcnn import Node
from tests.conftest import PKD_DIR, get_groundtruth

GT_RESULTS = get_groundtruth(Path(__file__).resolve())


@pytest.fixture
def mtcnn_config():
    with open(PKD_DIR / "configs" / "model" / "mtcnn.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = Path.cwd()

    return node_config


@pytest.fixture(
    params=[
        {"key": "min_size", "value": -0.5},
        {"key": "network_thresholds", "value": [-0.5, -0.5, -0.5]},
        {"key": "network_thresholds", "value": [1.5, 1.5, 1.5]},
        {"key": "scale_factor", "value": -0.5},
        {"key": "scale_factor", "value": 1.5},
        {"key": "score_threshold", "value": -0.5},
        {"key": "score_threshold", "value": 1.5},
    ],
)
def mtcnn_bad_config_value(request, mtcnn_config):
    mtcnn_config[request.param["key"]] = request.param["value"]
    return mtcnn_config


@pytest.mark.mlmodel
class TestMTCNN:
    def test_no_human_face_image(self, no_human_image, mtcnn_config):
        no_human_img = cv2.imread(no_human_image)
        mtcnn = Node(mtcnn_config)
        output = mtcnn.run({"img": no_human_img})
        expected_output = {
            "bboxes": np.empty((0, 4), dtype=np.float32),
            "bbox_scores": np.empty((0), dtype=np.float32),
            "bbox_labels": np.empty((0)),
        }
        assert output.keys() == expected_output.keys()
        npt.assert_equal(output["bboxes"], expected_output["bboxes"])
        npt.assert_equal(output["bbox_scores"], expected_output["bbox_scores"])
        npt.assert_equal(output["bbox_labels"], expected_output["bbox_labels"])

    def test_detect_face_bboxes(self, human_image, mtcnn_config):
        human_img = cv2.imread(human_image)
        mtcnn = Node(mtcnn_config)
        output = mtcnn.run({"img": human_img})

        assert "bboxes" in output
        assert output["bboxes"].size != 0

        image_name = Path(human_image).stem
        expected = GT_RESULTS[image_name]

        npt.assert_allclose(output["bboxes"], expected["bboxes"], atol=1e-3)
        npt.assert_equal(output["bbox_labels"], expected["bbox_labels"])
        npt.assert_allclose(output["bbox_scores"], expected["bbox_scores"], atol=1e-2)

    def test_invalid_config_value(self, mtcnn_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=mtcnn_bad_config_value)
        assert "must be" in str(excinfo.value)

    @mock.patch.object(WeightsDownloaderMixin, "_has_weights", return_value=True)
    def test_invalid_config_model_files(self, _, mtcnn_config):
        with pytest.raises(ValueError) as excinfo:
            mtcnn_config["weights"][mtcnn_config["model_format"]]["model_file"][
                mtcnn_config["model_type"]
            ] = "some/invalid/path"
            _ = Node(config=mtcnn_config)
        assert "Graph file does not exist. Please check that" in str(excinfo.value)
