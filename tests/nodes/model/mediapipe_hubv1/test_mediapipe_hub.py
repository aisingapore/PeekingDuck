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

import cv2
import pytest
import yaml

from peekingduck.nodes.model.mediapipe_hub import Node
from tests.conftest import PKD_DIR


@pytest.fixture
def mediapipe_hub_config():
    with open(PKD_DIR / "configs" / "model" / "mediapipe_hub.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = PKD_DIR

    return node_config


@pytest.mark.mlmodel
class TestMediaPipeHub:
    def test_require_subtask_config(self, mediapipe_hub_config):
        """Checks that the default `model_type: null` config value throws an error."""
        with pytest.raises(ValueError) as excinfo:
            _ = Node(mediapipe_hub_config)
        assert "subtask must be one of" in str(excinfo)

    @pytest.mark.parametrize(
        "config_value",
        [
            {"task": "object_detection", "subtask": "face", "model_type": 2},
            {"task": "pose_estimation", "subtask": "body", "model_type": 3},
        ],
    )
    def test_subtask_and_model_type_should_match(
        self, mediapipe_hub_config, config_value
    ):
        mediapipe_hub_config["task"] = config_value["task"]
        mediapipe_hub_config["subtask"] = config_value["subtask"]
        mediapipe_hub_config["model_type"] = config_value["model_type"]
        with pytest.raises(ValueError) as excinfo:
            _ = Node(mediapipe_hub_config)
        assert "model_type must be one of" in str(excinfo)

    def test_object_detection_empty_detections(
        self, create_image, mediapipe_hub_config
    ):
        mediapipe_hub_config["task"] = "object_detection"
        mediapipe_hub_config["subtask"] = "face"
        mediapipe_hub_config["model_type"] = 0
        img = create_image((416, 416, 3))
        mp_node = Node(mediapipe_hub_config)
        outputs = mp_node.run({"img": img})

        assert (
            len(outputs["bboxes"])
            == len(outputs["bbox_labels"])
            == len(outputs["bbox_scores"])
        )
        assert len(outputs["bboxes"]) == 0

    def test_pose_estimation_empty_detections(self, create_image, mediapipe_hub_config):
        mediapipe_hub_config["task"] = "pose_estimation"
        mediapipe_hub_config["subtask"] = "body"
        mediapipe_hub_config["model_type"] = 0
        img = create_image((416, 416, 3))
        mp_node = Node(mediapipe_hub_config)
        outputs = mp_node.run({"img": img})

        assert (
            len(outputs["bboxes"])
            == len(outputs["bbox_labels"])
            == len(outputs["keypoints"])
            == len(outputs["keypoint_conns"])
            == len(outputs["keypoint_scores"])
        )
        assert len(outputs["bboxes"]) == 0

    def test_object_detection_single_human(
        self, single_person_image, mediapipe_hub_config
    ):
        """Checks that inferencing on an image containing people produces some
        results.
        """
        mediapipe_hub_config["task"] = "object_detection"
        mediapipe_hub_config["subtask"] = "face"
        # model_type=0 is close-range and doesn't detect any faces
        mediapipe_hub_config["model_type"] = 1
        img = cv2.imread(single_person_image)
        mp_node = Node(mediapipe_hub_config)
        outputs = mp_node.run({"img": img})

        assert (
            len(outputs["bboxes"])
            == len(outputs["bbox_labels"])
            == len(outputs["bbox_scores"])
        )
        assert len(outputs["bboxes"]) > 0

    def test_pose_estimation_single_human(
        self, single_person_image, mediapipe_hub_config
    ):
        mediapipe_hub_config["task"] = "pose_estimation"
        mediapipe_hub_config["subtask"] = "body"
        mediapipe_hub_config["model_type"] = 1
        img = cv2.imread(single_person_image)
        mp_node = Node(mediapipe_hub_config)
        outputs = mp_node.run({"img": img})

        assert (
            len(outputs["bboxes"])
            == len(outputs["bbox_labels"])
            == len(outputs["keypoints"])
            == len(outputs["keypoint_conns"])
            == len(outputs["keypoint_scores"])
        )
        assert len(outputs["bboxes"]) == 1
