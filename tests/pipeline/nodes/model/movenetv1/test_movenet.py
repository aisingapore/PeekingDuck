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

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import yaml

from peekingduck.pipeline.nodes.model.movenet import Node
from tests.conftest import PKD_DIR, get_groundtruth

TOLERANCE = 1e-5
GT_RESULTS = get_groundtruth(Path(__file__).resolve())


@pytest.fixture
def movenet_config():
    with open(PKD_DIR / "configs" / "model" / "movenet.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = Path.cwd()

    return node_config


@pytest.fixture(
    params=[
        {"key": "bbox_score_threshold", "value": -0.5},
        {"key": "bbox_score_threshold", "value": 1.5},
        {"key": "keypoint_score_threshold", "value": -0.5},
        {"key": "keypoint_score_threshold", "value": 1.5},
        {"key": "model_type", "value": "bad_model_type"},
    ],
)
def movenet_bad_config_value(request, movenet_config):
    movenet_config[request.param["key"]] = request.param["value"]
    return movenet_config


@pytest.fixture(
    params=["singlepose_lightning", "singlepose_thunder", "multipose_lightning"]
)
def movenet_config_single(request, movenet_config):
    movenet_config["model_type"] = request.param
    return movenet_config


@pytest.fixture(params=["multipose_lightning"])
def movenet_config_multi(request, movenet_config):
    movenet_config["model_type"] = request.param
    return movenet_config


@pytest.mark.mlmodel
class TestMoveNet:
    def test_no_human_single(self, no_human_image, movenet_config_single):
        no_human_img = cv2.imread(no_human_image)
        model = Node(movenet_config_single)
        output = model.run({"img": no_human_img})
        expected_output = {
            "bboxes": np.zeros(0),
            "keypoints": np.zeros(0),
            "keypoint_scores": np.zeros(0),
            "keypoint_conns": np.zeros(0),
            "bbox_labels": np.zeros(0),
        }
        assert output.keys() == expected_output.keys()
        for i in expected_output.keys():
            npt.assert_array_equal(
                x=output[i],
                y=expected_output[i],
                err_msg=(
                    f"unexpected output for {i}, Expected {np.zeros(0)} got {output[i]}"
                ),
            )

    def test_no_human_multi(self, no_human_image, movenet_config_multi):
        no_human_img = cv2.imread(no_human_image)
        model = Node(movenet_config_multi)
        output = model.run({"img": no_human_img})
        expected_output = {
            "bboxes": np.zeros(0),
            "keypoints": np.zeros(0),
            "keypoint_scores": np.zeros(0),
            "keypoint_conns": np.zeros(0),
            "bbox_labels": np.zeros(0),
        }
        assert output.keys() == expected_output.keys()
        for i in expected_output.keys():
            npt.assert_array_equal(
                x=output[i],
                y=expected_output[i],
                err_msg=(
                    f"unexpected output for {i}, Expected {np.zeros(0)} got {output[i]}"
                ),
            )

    def test_single_person(self, single_person_image, movenet_config_single):
        single_person_img = cv2.imread(single_person_image)
        movenet = Node(movenet_config_single)
        output = movenet.run({"img": single_person_img})

        model_type = movenet.config["model_type"]
        image_name = Path(single_person_image).stem
        expected = GT_RESULTS[model_type][image_name]

        npt.assert_allclose(output["bboxes"], expected["bboxes"], atol=TOLERANCE)
        npt.assert_equal(output["bbox_labels"], expected["bbox_labels"])
        npt.assert_allclose(output["keypoints"], expected["keypoints"], atol=TOLERANCE)
        npt.assert_allclose(
            output["keypoint_conns"], expected["keypoint_conns"], atol=TOLERANCE
        )
        npt.assert_allclose(
            output["keypoint_scores"], expected["keypoint_scores"], atol=TOLERANCE
        )

    def test_multi_person(self, multi_person_image, movenet_config_multi):
        multi_person_img = cv2.imread(multi_person_image)
        movenet = Node(movenet_config_multi)
        output = movenet.run({"img": multi_person_img})

        model_type = movenet.config["model_type"]
        image_name = Path(multi_person_image).stem
        expected = GT_RESULTS[model_type][image_name]

        npt.assert_allclose(output["bboxes"], expected["bboxes"], atol=TOLERANCE)
        npt.assert_equal(output["bbox_labels"], expected["bbox_labels"])
        npt.assert_allclose(output["keypoints"], expected["keypoints"], atol=TOLERANCE)

        assert len(output["keypoint_conns"]) == len(expected["keypoint_conns"])
        # Detections can have different number of valid keypoint connections
        # and the keypoint connections result can be a ragged list lists.  When
        # converted to numpy array, the `keypoint_conns`` array will become
        # np.array([list(keypoint connections array), list(next keypoint
        # connections array), ...])
        # Thus, iterate through the detections
        for i, expected_keypoint_conns in enumerate(expected["keypoint_conns"]):
            npt.assert_allclose(
                output["keypoint_conns"][i],
                expected_keypoint_conns,
                atol=TOLERANCE,
            )

        npt.assert_allclose(
            output["keypoint_scores"], expected["keypoint_scores"], atol=TOLERANCE
        )

    def test_invalid_config_value(self, movenet_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=movenet_bad_config_value)
        assert "must be" in str(excinfo.value)
