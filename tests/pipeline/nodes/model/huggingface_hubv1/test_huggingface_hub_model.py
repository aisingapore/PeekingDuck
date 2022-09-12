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

from unittest import TestCase

import pytest
import yaml

from peekingduck.pipeline.nodes.model.huggingface_hubv1 import huggingface_hub_model
from tests.conftest import PKD_DIR, assert_msg_in_logs


@pytest.fixture
def huggingface_detr_config():
    with open(PKD_DIR / "configs" / "model" / "huggingface_hub.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = PKD_DIR
    node_config["model_type"] = "facebook/detr-resnet-50"

    return node_config


@pytest.mark.mlmodel
class TestObjectDetectionModel:
    def test_empty_detect_list(self, huggingface_detr_config):
        huggingface_detr_config["detect"] = []
        with TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.huggingface_hubv1."
            "huggingface_hub_model.ObjectDetectionModel"
        ) as captured:
            model = huggingface_hub_model.ObjectDetectionModel(huggingface_detr_config)

        assert_msg_in_logs(
            "`detect` list is empty, detecting all objects.", captured.records
        )
        assert len(model.detector.config.label2id) == len(model.detect_ids)

    def test_all_invalid_detect_label(self, huggingface_detr_config):
        huggingface_detr_config["detect"] = ["invalid_label_1", "invalid_label_2"]
        with TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.huggingface_hubv1."
            "huggingface_hub_model.ObjectDetectionModel"
        ) as captured:
            model = huggingface_hub_model.ObjectDetectionModel(huggingface_detr_config)

        assert_msg_in_logs("Invalid class names:", captured.records)
        assert_msg_in_logs("No valid entries", captured.records)
        assert len(model.detector.config.label2id) == len(model.detect_ids)

    def test_all_invalid_detect_id(self, huggingface_detr_config):
        # Assume no models use negative detect IDs
        huggingface_detr_config["detect"] = [-1, -2]
        with TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.huggingface_hubv1."
            "huggingface_hub_model.ObjectDetectionModel"
        ) as captured:
            model = huggingface_hub_model.ObjectDetectionModel(huggingface_detr_config)

        assert_msg_in_logs("Invalid detect IDs:", captured.records)
        assert_msg_in_logs("recommended to use class names", captured.records)
        assert_msg_in_logs("No valid entries", captured.records)
        assert len(model.detector.config.label2id) == len(model.detect_ids)

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
        model = huggingface_hub_model.ObjectDetectionModel(huggingface_detr_config)
        assert model.detect_ids == [0]
