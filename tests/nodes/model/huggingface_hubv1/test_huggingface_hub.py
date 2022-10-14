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

import os

import cv2
import pytest
import yaml

from peekingduck.nodes.model.huggingface_hub import Node
from tests.conftest import PKD_DIR, TEST_IMAGES_DIR


@pytest.fixture
def huggingface_hub_config():
    with open(PKD_DIR / "configs" / "model" / "huggingface_hub.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = PKD_DIR

    return node_config


@pytest.mark.mlmodel
class TestHuggingFaceHub:
    def test_require_model_type_config(self, huggingface_hub_config):
        """Checks that the default `model_type: null` config value throws an error."""
        with pytest.raises(ValueError) as excinfo:
            _ = Node(huggingface_hub_config)
        assert "model_type must be one of" in str(excinfo)

    @pytest.mark.parametrize(
        "config_value",
        [
            {"task": "instance_segmentation", "model_type": "facebook/detr-resnet-50"},
            {
                "task": "object_detection",
                "model_type": "facebook/detr-resnet-50-dc5-panoptic",
            },
        ],
    )
    def test_task_and_model_type_should_match(
        self, huggingface_hub_config, config_value
    ):
        huggingface_hub_config["task"] = config_value["task"]
        huggingface_hub_config["model_type"] = config_value["model_type"]
        with pytest.raises(ValueError) as excinfo:
            _ = Node(huggingface_hub_config)
        assert "model_type must be one of" in str(excinfo)

    def test_object_detection_empty_detections(
        self, create_image, huggingface_hub_config
    ):
        huggingface_hub_config["task"] = "object_detection"
        huggingface_hub_config["model_type"] = "facebook/detr-resnet-50"
        img = create_image((416, 416, 3))
        hf_node = Node(huggingface_hub_config)
        outputs = hf_node.run({"img": img})

        assert (
            len(outputs["bboxes"])
            == len(outputs["bbox_labels"])
            == len(outputs["bbox_scores"])
        )
        assert len(outputs["bboxes"]) == 0

    @pytest.mark.parametrize(
        "model_type", ["facebook/detr-resnet-50", "hustvl/yolos-tiny"]
    )
    def test_object_detection_single_human(
        self, single_person_image, huggingface_hub_config, model_type
    ):
        """Checks that inferencing on an image containing people produces some
        results. Uses both base model types: detr and yolos.
        """
        huggingface_hub_config["task"] = "object_detection"
        huggingface_hub_config["model_type"] = model_type
        img = cv2.imread(single_person_image)
        hf_node = Node(huggingface_hub_config)
        outputs = hf_node.run({"img": img})

        assert (
            len(outputs["bboxes"])
            == len(outputs["bbox_labels"])
            == len(outputs["bbox_scores"])
        )
        assert len(outputs["bboxes"]) > 0

    @pytest.mark.skipif(
        os.getenv("CI") is not None, reason="GitHub runner inadequate spec"
    )
    def test_instance_segmentation_empty_detections(self, huggingface_hub_config):
        huggingface_hub_config["task"] = "instance_segmentation"
        huggingface_hub_config["model_type"] = "facebook/detr-resnet-50-dc5-panoptic"
        img = cv2.imread(str(TEST_IMAGES_DIR / "black.jpg"))
        hf_node = Node(huggingface_hub_config)
        outputs = hf_node.run({"img": img})

        print(outputs["bbox_labels"])
        assert (
            len(outputs["bboxes"])
            == len(outputs["bbox_labels"])
            == len(outputs["bbox_scores"])
            == len(outputs["masks"])
        )
        assert len(outputs["bboxes"]) == 0

    @pytest.mark.parametrize(
        "model_type",
        ["facebook/detr-resnet-50-dc5-panoptic", "facebook/maskformer-swin-tiny-ade"],
    )
    @pytest.mark.skipif(
        os.getenv("CI") is not None, reason="GitHub runner inadequate spec"
    )
    def test_instance_segmentation_single_human(
        self, single_person_image, huggingface_hub_config, model_type
    ):
        """Checks that inferencing on an image containing people produces some
        results. Uses both base model types: detr and yolos.
        """
        huggingface_hub_config["task"] = "instance_segmentation"
        huggingface_hub_config["model_type"] = model_type
        img = cv2.imread(single_person_image)
        hf_node = Node(huggingface_hub_config)
        outputs = hf_node.run({"img": img})

        assert (
            len(outputs["bboxes"])
            == len(outputs["bbox_labels"])
            == len(outputs["bbox_scores"])
            == len(outputs["masks"])
        )
        assert len(outputs["bboxes"]) > 0
