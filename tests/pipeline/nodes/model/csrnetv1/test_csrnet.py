"""
Copyright 2021 AI Singapore

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pathlib import Path
from unittest import TestCase, mock

import cv2
import math
import numpy as np
import pytest
import yaml

from peekingduck.pipeline.nodes.model.csrnet import Node
from peekingduck.pipeline.nodes.model.csrnetv1.csrnet_files.predictor import Predictor


@pytest.fixture
def csrnet_config():
    filepath = (
        Path.cwd()
        / "tests"
        / "pipeline"
        / "nodes"
        / "model"
        / "csrnetv1"
        / "test_csrnet.yml"
    )
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


@pytest.fixture(params=["sparse", "dense"])
def csrnet(request, csrnet_config):
    csrnet_config["model_type"] = request.param
    node = Node(csrnet_config)

    return node


@pytest.fixture()
def csrnet_predictor(csrnet_config, model_dir):
    predictor = Predictor(csrnet_config, model_dir)

    return predictor


def replace_download_weights(model_dir, blob_file):
    return False


@pytest.mark.mlmodel
class TestCsrnet:
    def test_no_human(self, test_no_human_images, csrnet):
        blank_image = cv2.imread(test_no_human_images)
        output = csrnet.run({"img": blank_image})
        assert list(output.keys()) == ["density_map", "count"]
        assert math.ceil(np.sum(output["density_map"])) == output["count"]
        assert output["count"] < 9

    def test_crowd(self, test_crowd_images, csrnet):
        crowd_image = cv2.imread(test_crowd_images)
        output = csrnet.run({"img": crowd_image})
        assert list(output.keys()) == ["density_map", "count"]
        assert math.ceil(np.sum(output["density_map"])) == output["count"]
        assert output["count"] >= 10

    def test_no_weights(self, csrnet_config):
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
