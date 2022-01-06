# Copyright 2021 AI Singapore
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


@pytest.fixture
def hrnet_config():
    with open(Path(__file__).resolve().parent / "test_hrnet.yml") as file:
        node_config = yaml.safe_load(file)
    node_config["root"] = Path.cwd()

    return node_config


@pytest.mark.mlmodel
class TestHrnet:
    def test_no_human_image(self, test_no_human_images, hrnet_config):
        """Tests HRnet on images with no humans present."""
        blank_image = cv2.imread(test_no_human_images)
        hrnet = Node(hrnet_config)
        output = hrnet.run({"img": blank_image, "bboxes": np.empty((0, 4))})
        expected_output = {
            "keypoints": np.zeros(0),
            "keypoint_scores": np.zeros(0),
            "keypoint_conns": np.zeros(0),
        }

        assert output.keys() == expected_output.keys(), "missing keys"
        for i in expected_output.keys():
            npt.assert_array_equal(
                output[i], expected_output[i]
            ), "unexpected output for {}".format(i)

    def test_return_at_least_one_person_and_one_bbox(
        self, test_human_images, hrnet_config
    ):
        """Tests HRnet on images with at least one human present. Bbox
        coordinates is set as the entire image.
        """
        test_img = cv2.imread(test_human_images)
        img_h, img_w, _ = test_img.shape
        hrnet = Node(hrnet_config)
        output = hrnet.run(
            {"img": test_img, "bboxes": np.array([[0, 0, img_w, img_h]])}
        )

        assert "keypoints" in output
        assert "keypoint_scores" in output
        assert "keypoint_conns" in output
        assert output["keypoints"].size != 0
