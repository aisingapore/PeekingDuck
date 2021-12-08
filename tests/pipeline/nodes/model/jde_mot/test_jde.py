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
import numpy as np
import numpy.testing as npt
import pytest
import yaml

from peekingduck.pipeline.nodes.model.jde import Node
from peekingduck.weights_utils.finder import PEEKINGDUCK_WEIGHTS_SUBDIR


@pytest.fixture
def jde_config():
    file_path = (
        Path.cwd()
        / "tests"
        / "pipeline"
        / "nodes"
        / "model"
        / "jde_mot"
        / "test_jde.yml"
    )
    with open(file_path) as file:
        node_config = yaml.safe_load(file)
    node_config["root"] = Path.cwd()

    return node_config


@pytest.fixture
def jde(jde_config):
    with mock.patch("torch.cuda.is_available", return_value=False):
        node = Node(jde_config)
        return node


@pytest.fixture(
    params=[
        {"key": "iou_threshold", "value": -0.5},
        {"key": "iou_threshold", "value": 1.5},
        {"key": "conf_threshold", "value": -0.5},
        {"key": "conf_threshold", "value": 1.5},
        {"key": "nms_threshold", "value": -0.5},
        {"key": "nms_threshold", "value": 1.5},
    ],
)
def jde_bad_config_value(request, jde_config):
    jde_config[request.param["key"]] = request.param["value"]
    return jde_config


def replace_download_weights(*_):
    pass


@pytest.mark.mlmodel
class TestJDE:
    def test_no_human_image(self, test_no_human_images, jde):
        blank_image = cv2.imread(test_no_human_images)
        output = jde.run({"img": blank_image})
        expected_output = {
            "bboxes": [],
            "obj_tags": [],
            "bbox_labels": [],
        }
        assert output.keys() == expected_output.keys()
        npt.assert_equal(output["bboxes"], expected_output["bboxes"])
        npt.assert_equal(output["obj_tags"], expected_output["obj_tags"])
        npt.assert_equal(output["bbox_labels"], expected_output["bbox_labels"])

    def test_detect_human_bboxes(self, test_human_videos, jde):
        # Create a VideoCapture object and read from input file
        cap = cv2.VideoCapture(test_human_videos)
        # Read until video is completed
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                output = jde.run({"img": frame})
                if output["bboxes"]:
                    assert len(output["bboxes"]) == 3
                    assert len(output["bbox_labels"]) == 3
            else:
                break
        # When everything done, release the video capture object
        cap.release()

    def test_invalid_config_value(self, jde_bad_config_value):
        with pytest.raises(ValueError):
            _ = Node(config=jde_bad_config_value)

    def test_invalid_config_model_files(self, jde_config):
        with mock.patch(
            "peekingduck.weights_utils.checker.has_weights", return_value=True
        ), pytest.raises(FileNotFoundError):
            jde_config["weights"]["model_file"]["jde"] = "some/invalid/path"
            _ = Node(config=jde_config)

    def test_invalid_image(self, test_no_human_images, jde):
        blank_image = cv2.imread(test_no_human_images)
        # Potentially passing in a file path or a tuple from image reader
        # output
        with pytest.raises(AttributeError):
            _ = jde.run({"img": Path.cwd()})
        with pytest.raises(AttributeError):
            _ = jde.run({"img": ("image name", blank_image)})

    def test_no_weights(self, jde_config):
        weights_dir = jde_config["root"].parent / PEEKINGDUCK_WEIGHTS_SUBDIR
        with mock.patch(
            "peekingduck.weights_utils.checker.has_weights", return_value=False
        ), mock.patch(
            "peekingduck.weights_utils.downloader.download_weights",
            wraps=replace_download_weights,
        ), TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.jde_mot.jde_model.logger"
        ) as captured:
            jde = Node(config=jde_config)
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
            assert jde is not None
