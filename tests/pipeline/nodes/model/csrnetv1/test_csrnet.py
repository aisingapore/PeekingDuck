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
from unittest import TestCase, mock

import cv2
import pytest
import yaml

from peekingduck.pipeline.nodes.base import (
    PEEKINGDUCK_WEIGHTS_SUBDIR,
    WeightsDownloaderMixin,
)
from peekingduck.pipeline.nodes.model.csrnet import Node
from tests.conftest import PKD_DIR


@pytest.fixture(params=["sparse", "dense"])
def csrnet_config():
    with open(PKD_DIR / "configs" / "model" / "csrnet.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = Path.cwd()

    return node_config


@pytest.fixture(
    params=[{"key": "width", "value": -1}, {"key": "width", "value": 0}],
)
def csrnet_bad_config_value(request, csrnet_config):
    csrnet_config[request.param["key"]] = request.param["value"]
    return csrnet_config


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
        weights_dir = csrnet_config["root"].parent / PEEKINGDUCK_WEIGHTS_SUBDIR
        with mock.patch.object(
            WeightsDownloaderMixin, "_has_weights", return_value=False
        ), mock.patch.object(
            WeightsDownloaderMixin, "_download_blob_to", wraps=replace_download_weights
        ), mock.patch.object(
            WeightsDownloaderMixin, "extract_file", wraps=replace_download_weights
        ), TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.csrnetv1.csrnet_model.logger"
        ) as captured:
            csrnet = Node(config=csrnet_config)
            # records 0 - 20 records are updates to configs
            assert (
                captured.records[0].getMessage()
                == "No weights detected. Proceeding to download..."
            )
            assert (
                captured.records[1].getMessage()
                == f"Weights downloaded to {weights_dir}."
            )
            assert csrnet is not None

    def test_invalid_config_value(self, csrnet_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=csrnet_bad_config_value)
        assert "must be more than 0" in str(excinfo.value)
