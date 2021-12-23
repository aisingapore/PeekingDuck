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
import yaml
import pytest
import numpy as np
import cv2
from peekingduck.pipeline.nodes.model.osnet import Node
from peekingduck.pipeline.nodes.model.osnetv1.osnet_files.build_model import build_model
from peekingduck.weights_utils.finder import PEEKINGDUCK_WEIGHTS_SUBDIR


@pytest.fixture
def size():
    return (400, 600, 3)


@pytest.fixture
def osnet_config():
    filepath = (
        Path.cwd()
        / "tests"
        / "pipeline"
        / "nodes"
        / "model"
        / "osnetv1"
        / "test_osnet.yml"
    )
    with open(filepath) as file:
        node_config = yaml.safe_load(file)
    node_config["root"] = Path.cwd()
    node_config["query_root_dir"] = (
        str(node_config["root"]) + node_config["query_root_dir"]
    )
    return node_config


@pytest.fixture()
def osnet(osnet_config):
    node = Node(osnet_config)
    return node


def replace_download_weights(*_):
    pass


@pytest.mark.mlmodel
class TestOSNet:
    def test_no_queries(self, create_image, size, osnet):
        img1 = create_image(size)
        array1 = []
        input1 = {
            "img": img1,
            "bboxes": array1,
        }
        assert osnet.run(input1)["obj_tags"] == []
        np.testing.assert_equal(input1["img"], img1)
        np.testing.assert_equal(input1["bboxes"], array1)

    def test_single_query(self, test_human_images, osnet):
        test_img = cv2.imread(test_human_images)
        array1 = np.array([[0.16938463, 0.06940075, 0.5769387, 0.904104]])
        output = osnet.run(
            {
                "img": test_img,
                "bboxes": array1,
            }
        )
        assert len(output["bboxes"]) == 1
        assert len(output["obj_tags"]) == 1

    def test_osnet_model_initialization(self):
        model = build_model("osnet_x1_0", num_classes=1)
        assert model is not None

    def test_no_weights(self, osnet_config):
        weights_dir = osnet_config["root"].parent / PEEKINGDUCK_WEIGHTS_SUBDIR
        with mock.patch(
            "peekingduck.weights_utils.checker.has_weights", return_value=False
        ), mock.patch(
            "peekingduck.weights_utils.downloader.download_weights",
            wraps=replace_download_weights,
        ), TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.osnetv1.osnet_model.logger"
        ) as captured:
            osnet = Node(config=osnet_config)
            print(captured)
            # records 0 - 20 records are updates to configs
            assert (
                captured.records[0].getMessage()
                == "No weights detected. Proceeding to download..."
            )
            assert (
                captured.records[1].getMessage()
                == f"Weights downloaded to {weights_dir}."
            )
            assert osnet is not None
