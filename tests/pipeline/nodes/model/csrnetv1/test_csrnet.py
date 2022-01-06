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
import pytest
import yaml

from peekingduck.pipeline.nodes.model.csrnet import Node
from peekingduck.pipeline.nodes.model.csrnetv1.csrnet_files.predictor import Predictor


@pytest.fixture(params=["sparse", "dense"])
def csrnet_config():
    filepath = Path(__file__).resolve().parent / "test_csrnet.yml"
    with open(filepath) as file:
        node_config = yaml.safe_load(file)
    node_config["root"] = Path.cwd()

    return node_config


@pytest.fixture
def model_dir(csrnet_config):
    return (
        csrnet_config["root"].parent
        / "peekingduck_weights"
        / csrnet_config["weights"]["model_subdir"]
    )


@pytest.mark.mlmodel
class TestCsrnet:
    def test_no_human(self, test_no_human_images, csrnet_config):
        blank_image = cv2.imread(test_no_human_images)
        csrnet = Node(csrnet_config)
        output = csrnet.run({"img": blank_image})
        assert list(output.keys()) == ["density_map", "count"]
        # Model is less accurate and detects extra people when cnt is low or
        # none. Threshold of 9 is chosen based on the min cnt in ShanghaiTech
        # dataset
        assert output["count"] < 9

    def test_crowd(self, test_crowd_images, csrnet_config):
        crowd_image = cv2.imread(test_crowd_images)
        csrnet = Node(csrnet_config)
        output = csrnet.run({"img": crowd_image})
        assert list(output.keys()) == ["density_map", "count"]
        assert output["count"] >= 10

    def test_no_weights(self, csrnet_config, replace_download_weights):
        with mock.patch(
            "peekingduck.weights_utils.checker.has_weights", return_value=False
        ), mock.patch(
            "peekingduck.weights_utils.downloader.download_weights",
            wraps=replace_download_weights,
        ), TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.csrnetv1.csrnet_model.logger"
        ) as captured:
            csrnet = Node(config=csrnet_config)
            # records 0 - 20 records are updates to configs
            assert (
                captured.records[0].getMessage()
                == "---no weights detected. proceeding to download...---"
            )
            assert "weights downloaded" in captured.records[1].getMessage()
            assert csrnet is not None

    def test_model_initialization(self, csrnet_config, model_dir):
        predictor = Predictor(csrnet_config, model_dir)
        model = predictor.csrnet
        assert model is not None
