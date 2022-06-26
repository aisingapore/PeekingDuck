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

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import yaml

from peekingduck.pipeline.nodes.model.hrnet import Node
from tests.conftest import PKD_DIR, get_groundtruth

TOLERANCE = 1e-5
GT_RESULTS = get_groundtruth(Path(__file__).resolve())


@pytest.fixture
def hrnet_config():
    with open(PKD_DIR / "configs" / "model" / "hrnet.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = Path.cwd()

    return node_config


@pytest.fixture(
    params=[
        {"key": "score_threshold", "value": -0.5},
        {"key": "score_threshold", "value": 1.5},
    ],
)
def hrnet_bad_config_value(request, hrnet_config):
    hrnet_config[request.param["key"]] = request.param["value"]
    return hrnet_config


@pytest.mark.mlmodel
class TestHrnet:
    def test_no_human_image(self, no_human_image, hrnet_config):
        """Tests HRnet on images with no humans present."""
        no_human_img = cv2.imread(no_human_image)
        hrnet = Node(hrnet_config)
        output = hrnet.run({"img": no_human_img, "bboxes": np.empty((0, 4))})
        expected_output = {
            "keypoints": np.zeros(0),
            "keypoint_scores": np.zeros(0),
            "keypoint_conns": np.zeros(0),
        }

        assert output.keys() == expected_output.keys(), "missing keys"
        for i in expected_output.keys():
            npt.assert_array_equal(
                output[i], expected_output[i], err_msg=f"unexpected output for {i}"
            )

    def test_single_human(self, single_person_image, hrnet_config):
        """Using bboxes from MoveNet multipose_thunder."""
        single_human_img = cv2.imread(single_person_image)
        hrnet = Node(hrnet_config)
        output = hrnet.run(
            {
                "img": single_human_img,
                "bboxes": np.array([[0.181950, 0.081390, 0.590675, 0.902916]]),
            }
        )

        model_type = hrnet.config["model_type"]
        image_name = Path(single_person_image).stem
        expected = GT_RESULTS[model_type][image_name]

        npt.assert_allclose(output["keypoints"], expected["keypoints"], atol=TOLERANCE)
        npt.assert_allclose(
            output["keypoint_conns"], expected["keypoint_conns"], atol=TOLERANCE
        )
        npt.assert_allclose(
            output["keypoint_scores"], expected["keypoint_scores"], atol=TOLERANCE
        )

    def test_multi_person(self, multi_person_image, hrnet_config):
        """Using bboxes from MoveNet multipose_thunder."""
        multi_person_img = cv2.imread(multi_person_image)
        hrnet = Node(hrnet_config)

        model_type = hrnet.config["model_type"]
        image_name = Path(multi_person_image).stem
        expected = GT_RESULTS[model_type][image_name]
        output = hrnet.run(
            {"img": multi_person_img, "bboxes": np.array(expected["bboxes"])}
        )

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

    def test_invalid_config_value(self, hrnet_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=hrnet_bad_config_value)
        assert "_threshold must be between [0.0, 1.0]" in str(excinfo.value)
