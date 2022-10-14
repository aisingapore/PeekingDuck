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
from unittest import TestCase

import pytest
import yaml

from peekingduck.nodes.model.huggingface_hubv1.models import (
    instance_segmentation,
    object_detection,
)
from tests.conftest import PKD_DIR, assert_msg_in_logs


@pytest.fixture
def huggingface_detr_config():
    with open(PKD_DIR / "configs" / "model" / "huggingface_hub.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = PKD_DIR
    node_config["task"] = "object_detection"
    node_config["model_type"] = "facebook/detr-resnet-50"

    return node_config


@pytest.fixture(
    params=["facebook/detr-resnet-50-dc5-panoptic", "facebook/maskformer-swin-tiny-ade"]
)
def huggingface_segmentation_config(request):
    """Tests both PanopticSegmenter and InstanceSegmenter."""
    with open(PKD_DIR / "configs" / "model" / "huggingface_hub.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = PKD_DIR
    node_config["task"] = "instance_segmentation"
    node_config["model_type"] = request.param

    return node_config


@pytest.mark.mlmodel
class TestObjectDetectionModel:
    def test_empty_detect_list(self, huggingface_detr_config):
        huggingface_detr_config["detect"] = []
        with TestCase.assertLogs(
            "peekingduck.nodes.model.huggingface_hubv1."
            "object_detection.ObjectDetectionModel"
        ) as captured:
            model = object_detection.ObjectDetectionModel(huggingface_detr_config)
            model.post_init()

        assert_msg_in_logs(
            "`detect` list is empty, detecting all objects.", captured.records
        )
        assert len(model.adaptor.model.config.label2id) == len(model.detect_ids)

    def test_all_invalid_detect_label(self, huggingface_detr_config):
        huggingface_detr_config["detect"] = ["invalid_label_1", "invalid_label_2"]
        with TestCase.assertLogs(
            "peekingduck.nodes.model.huggingface_hubv1."
            "object_detection.ObjectDetectionModel"
        ) as captured:
            model = object_detection.ObjectDetectionModel(huggingface_detr_config)
            model.post_init()

        assert_msg_in_logs("Invalid class names:", captured.records)
        assert_msg_in_logs("No valid entries", captured.records)
        assert len(model.adaptor.model.config.label2id) == len(model.detect_ids)

    def test_all_invalid_detect_id(self, huggingface_detr_config):
        # Assume no models use negative detect IDs
        huggingface_detr_config["detect"] = [-1, -2]
        with TestCase.assertLogs(
            "peekingduck.nodes.model.huggingface_hubv1."
            "object_detection.ObjectDetectionModel"
        ) as captured:
            model = object_detection.ObjectDetectionModel(huggingface_detr_config)
            model.post_init()

        assert_msg_in_logs("Invalid detect IDs:", captured.records)
        assert_msg_in_logs("recommended to use class names", captured.records)
        assert_msg_in_logs("No valid entries", captured.records)
        assert len(model.adaptor.model.config.label2id) == len(model.detect_ids)

    def test_mixed_detect_id_and_label(self, huggingface_detr_config):
        # Assume no models use negative detect IDs. This should remove all
        # `detect` entries except for ID 0.
        huggingface_detr_config["detect"] = [
            "invalid_label_1",
            "invalid_label_2",
            -1,
            -2,
            0,
        ]
        model = object_detection.ObjectDetectionModel(huggingface_detr_config)
        model.post_init()
        assert model.detect_ids == [0]


@pytest.mark.mlmodel
@pytest.mark.skipif(os.getenv("CI") is not None, reason="GitHub runner inadequate spec")
class TestInstanceSegmentationModel:
    def test_empty_detect_list(self, huggingface_segmentation_config):
        huggingface_segmentation_config["detect"] = []
        with TestCase.assertLogs(
            "peekingduck.nodes.model.huggingface_hubv1."
            "instance_segmentation.InstanceSegmentationModel"
        ) as captured:
            model = instance_segmentation.InstanceSegmentationModel(
                huggingface_segmentation_config
            )
            model.post_init()

        assert_msg_in_logs(
            "`detect` list is empty, detecting all objects.", captured.records
        )
        assert len(model.adaptor.model.config.label2id) == len(model.detect_ids)

    def test_all_invalid_detect_label(self, huggingface_segmentation_config):
        huggingface_segmentation_config["detect"] = [
            "invalid_label_1",
            "invalid_label_2",
        ]
        with TestCase.assertLogs(
            "peekingduck.nodes.model.huggingface_hubv1."
            "instance_segmentation.InstanceSegmentationModel"
        ) as captured:
            model = instance_segmentation.InstanceSegmentationModel(
                huggingface_segmentation_config
            )
            model.post_init()

        assert_msg_in_logs("Invalid class names:", captured.records)
        assert_msg_in_logs("No valid entries", captured.records)
        assert len(model.adaptor.model.config.label2id) == len(model.detect_ids)

    def test_all_invalid_detect_id(self, huggingface_segmentation_config):
        # Assume no models use negative detect IDs
        huggingface_segmentation_config["detect"] = [-1, -2]
        with TestCase.assertLogs(
            "peekingduck.nodes.model.huggingface_hubv1."
            "instance_segmentation.InstanceSegmentationModel"
        ) as captured:
            model = instance_segmentation.InstanceSegmentationModel(
                huggingface_segmentation_config
            )
            model.post_init()

        assert_msg_in_logs("Invalid detect IDs:", captured.records)
        assert_msg_in_logs("recommended to use class names", captured.records)
        assert_msg_in_logs("No valid entries", captured.records)
        assert len(model.adaptor.model.config.label2id) == len(model.detect_ids)

    def test_mixed_detect_id_and_label(self, huggingface_segmentation_config):
        # Assume no models use negative detect IDs. This should remove all
        # `detect` entries except for ID 0.
        huggingface_segmentation_config["detect"] = [
            "invalid_label_1",
            "invalid_label_2",
            -1,
            -2,
            0,
        ]
        model = instance_segmentation.InstanceSegmentationModel(
            huggingface_segmentation_config
        )
        model.post_init()
        assert model.detect_ids == [0]
