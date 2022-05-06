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

from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import yaml

from peekingduck.pipeline.nodes.base import WeightsDownloaderMixin
from peekingduck.pipeline.nodes.model.posenet import Node
from tests.conftest import PKD_DIR, get_groundtruth

TOLERANCE = 1e-5
GT_RESULTS = get_groundtruth(Path(__file__).resolve())


@pytest.fixture
def posenet_config():
    with open(PKD_DIR / "configs" / "model" / "posenet.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = Path.cwd()
    return node_config


@pytest.fixture(params=[50, 75, 100, "resnet"])
def posenet_type(request, posenet_config):
    posenet_config["model_type"] = request.param
    return posenet_config


@pytest.fixture(
    params=[
        {"key": "score_threshold", "value": -0.5},
        {"key": "score_threshold", "value": 1.5},
        {"key": "model_type", "value": 101},
        {"key": "model_type", "value": "inception"},
    ],
)
def posenet_bad_config_value(request, posenet_config):
    posenet_config[request.param["key"]] = request.param["value"]
    return posenet_config


@pytest.mark.mlmodel
class TestPoseNet:
    def test_no_detection(self, no_human_image, posenet_type):
        no_human_img = cv2.imread(str(no_human_image))
        posenet = Node(posenet_type)
        output = posenet.run({"img": no_human_img})
        expected_output = {
            "bboxes": np.zeros(0),
            "keypoints": np.zeros(0),
            "keypoint_scores": np.zeros(0),
            "keypoint_conns": np.zeros(0),
            "bbox_labels": np.zeros(0),
        }
        assert output.keys() == expected_output.keys(), "missing keys"
        for i in expected_output.keys():
            npt.assert_array_equal(
                output[i], expected_output[i], err_msg=f"unexpected output for {i}"
            )

    def test_different_models(self, human_image, posenet_type):
        human_img = cv2.imread(str(human_image))
        posenet = Node(posenet_type)
        output = posenet.run({"img": human_img})
        expected_output = dict.fromkeys(
            ["bboxes", "keypoints", "keypoint_scores", "keypoint_conns", "bbox_labels"]
        )
        assert output.keys() == expected_output.keys(), "missing keys"
        for i in expected_output.keys():
            assert len(output[i]) >= 1, "unexpected number of outputs for {}".format(i)
        for label in output["bbox_labels"]:
            assert label == "person"

    def test_single_human(self, single_person_image, posenet_type):
        single_human_img = cv2.imread(single_person_image)
        posenet = Node(posenet_type)
        output = posenet.run({"img": single_human_img})

        model_type = posenet.config["model_type"]
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

    def test_multi_person(self, multi_person_image, posenet_type):
        multi_person_img = cv2.imread(multi_person_image)
        posenet = Node(posenet_type)
        output = posenet.run({"img": multi_person_img})

        model_type = posenet.config["model_type"]
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

    def test_detect_specified_max_poses(self, multi_person_image, posenet_type):
        multi_person_img = cv2.imread(multi_person_image)

        posenet_type["max_pose_detection"] = 1
        posenet = Node(posenet_type)
        output = posenet.run({"img": multi_person_img})

        # only check length of outputs as groundtruth is tested in another test
        assert len(output["bboxes"]) == 1
        assert len(output["bbox_labels"]) == 1
        assert len(output["keypoints"]) == 1

        assert len(output["keypoint_conns"]) == 1
        assert len(output["keypoint_scores"]) == 1

    def test_invalid_config_value(self, posenet_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=posenet_bad_config_value)
        assert "must be" in str(excinfo.value)

    @mock.patch.object(WeightsDownloaderMixin, "_has_weights", return_value=True)
    def test_invalid_config_model_files(self, _, posenet_config):
        with pytest.raises(ValueError) as excinfo:
            posenet_config["weights"][posenet_config["model_format"]]["model_file"][
                posenet_config["model_type"]
            ] = "some/invalid/path"
            _ = Node(config=posenet_config)
        assert "Graph file does not exist. Please check that" in str(excinfo.value)
