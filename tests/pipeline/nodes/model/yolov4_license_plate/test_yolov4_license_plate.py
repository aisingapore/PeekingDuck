# (26.10.21) Work in progress to troubleshoot this unit test, where testing
# only fails in Github runner

from pathlib import Path
from unittest import TestCase, mock

import cv2
import numpy as np
import pytest
import yaml

from peekingduck.pipeline.nodes.model.yolo_license_plate import Node
from peekingduck.pipeline.nodes.model.yolov4_license_plate.licenseplate_files.detector import (
    Detector,
)


@pytest.fixture
def yolo_config():
    filepath = (
        Path.cwd()
        / "tests"
        / "pipeline"
        / "nodes"
        / "model"
        / "yolov4_license_plate"
        / "test_yolov4_license_plate.yml"
    )
    with open(filepath) as file:
        node_config = yaml.safe_load(file)
    node_config["root"] = Path.cwd()

    return node_config


@pytest.fixture
def model_dir(yolo_config):
    return (
        yolo_config["root"].parent
        / "peekingduck_weights"
        / yolo_config["weights"]["model_subdir"]
    )


@pytest.fixture(params=["v4", "v4tiny"])
def yolo(request, yolo_config):
    yolo_config["model_type"] = request.param
    node = Node(yolo_config)

    return node


@pytest.fixture()
def yolo_detector(yolo_config):
    yolo_config["model_type"] = "v4tiny"
    detector = Detector(yolo_config)

    return detector


def replace_download_weights(model_dir, blob_file):
    return False


@pytest.mark.mlmodel
class TestLPYolo:
    def test_no_lp_image(self, test_no_lp_images, yolo):
        blank_image = cv2.imread(test_no_lp_images)
        output = yolo.run({"img": blank_image})
        expected_output = {"bboxes": [], "bbox_labels": [], "bbox_scores": []}
        assert output.keys() == expected_output.keys()
        assert type(output["bboxes"]) == np.ndarray
        assert type(output["bbox_labels"]) == np.ndarray
        assert type(output["bbox_scores"]) == np.ndarray
        assert len(output["bboxes"]) == 0
        assert len(output["bbox_labels"]) == 0
        assert len(output["bbox_scores"]) == 0

    def test_at_least_one_lp_image(self, test_lp_images, yolo):
        test_img = cv2.imread(test_lp_images)
        output = yolo.run({"img": test_img})
        assert "bboxes" in output
        assert len(output["bboxes"]) != 0
        assert len(output["bboxes"]) == len(output["bbox_labels"])

    def test_no_weights(self, yolo_config):
        with mock.patch(
            "peekingduck.weights_utils.checker.has_weights", return_value=False
        ), mock.patch(
            "peekingduck.weights_utils.downloader.download_weights",
            wraps=replace_download_weights,
        ), TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.yolov4_license_plate.LP_detector_model.logger"
        ) as captured:
            yolo = Node(config=yolo_config)
            # records 0 - 20 records are updates to configs
            assert (
                captured.records[0].getMessage()
                == "---no yolo license plate weights detected. proceeding to download...---"
            )
            assert "weights downloaded" in captured.records[1].getMessage()
            assert yolo is not None

    def test_model_initialization(self, yolo_config, model_dir):
        detector = Detector(yolo_config, model_dir)
        model = detector.yolo
        assert model is not None
