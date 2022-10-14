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

"""
Test for draw poses node
"""

from unittest import mock

import cv2
import numpy as np
import pytest
import yaml

from peekingduck.nodes.draw.poses import Node
from tests.conftest import PKD_DIR


@pytest.fixture
def draw_poses_config():
    with open(PKD_DIR / "configs" / "draw" / "poses.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = PKD_DIR
    return node_config


@pytest.fixture(
    params=[
        {"value": "whiteee"},
        {"value": "123"},
    ]
)
def invalid_color_name(request):
    return request.param["value"]


@pytest.fixture(
    params=[
        {"value": [-1, 255, 255]},
        {"value": [256, 255, 255]},
    ]
)
def invalid_color_range(request):
    return request.param["value"]


@pytest.fixture(
    params=[
        {"value": [0, 0, 0, 0]},
        {"value": [0, 0]},
    ]
)
def invalid_color_tuple_len(request):
    return request.param["value"]


class TestPoses:
    def test_no_poses(self, draw_poses_config, create_image):
        draw_poses_node = Node(draw_poses_config)

        original_img = create_image((28, 28, 3))
        output_img = original_img.copy()

        keypoints = np.empty((0, 17, 2))  # keypoints shape
        keypoint_scores = np.empty((0, 17))  # keypoint_scores shape
        keypoint_conns = np.empty((0, 15, 2, 2))  # keypoint_conns shape

        inputs = {
            "keypoints": keypoints,
            "keypoint_scores": keypoint_scores,
            "keypoint_conns": keypoint_conns,
            "img": output_img,
        }

        draw_poses_node.run(inputs)

        np.testing.assert_equal(original_img, output_img)

    @pytest.mark.parametrize(
        "config_key", ["keypoint_dot_color", "keypoint_connect_color"]
    )
    def test_invalid_color_name(
        self, config_key, draw_poses_config, invalid_color_name
    ):
        draw_poses_config[config_key] = invalid_color_name

        with pytest.raises(ValueError) as excinfo:
            _ = Node(draw_poses_config)
        assert f"{config_key} must be one of" in str(excinfo.value)

    @pytest.mark.parametrize(
        "config_key", ["keypoint_dot_color", "keypoint_connect_color"]
    )
    def test_invalid_color_range(
        self, config_key, draw_poses_config, invalid_color_range
    ):
        draw_poses_config[config_key] = invalid_color_range

        with pytest.raises(ValueError) as excinfo:
            _ = Node(draw_poses_config)
        assert f"All elements of {config_key} must be between [0.0, 255.0]" in str(
            excinfo.value
        )

    @pytest.mark.parametrize(
        "config_key", ["keypoint_dot_color", "keypoint_connect_color"]
    )
    def test_invalid_color_tuple_len(
        self, config_key, draw_poses_config, invalid_color_tuple_len
    ):
        draw_poses_config[config_key] = invalid_color_tuple_len

        with pytest.raises(ValueError) as excinfo:
            _ = Node(draw_poses_config)
        assert "BGR values must be a list of length 3." in str(excinfo.value)

    @mock.patch("cv2.line", wraps=cv2.line)
    @mock.patch("cv2.circle", wraps=cv2.circle)
    def test_if_configs_are_used(
        self, mock_circle, mock_line, draw_poses_config, create_image
    ):
        color = [10, 20, 30]  # pass in a color that is different from the default
        draw_poses_config.update(
            {"keypoint_dot_color": color, "keypoint_connect_color": color}
        )
        draw_poses_node = Node(draw_poses_config)

        original_img = create_image((28, 28, 3))
        output_img = original_img.copy()

        keypoints = np.random.rand(1, 17, 2)  # keypoints shape for 1 person
        keypoint_scores = np.random.rand(1, 17)  # keypoint_scores shape for 1 person
        keypoint_conns = np.random.rand(
            1, 15, 2, 2
        )  # keypoint_conns shape for 1 person

        inputs = {
            "keypoints": keypoints,
            "keypoint_scores": keypoint_scores,
            "keypoint_conns": keypoint_conns,
            "img": output_img,
        }

        draw_poses_node.run(inputs)

        # asserts that cv2.circle()'s 4th argument `color` is the same as the config's color
        assert mock_circle.call_args[0][3] == tuple(color)
        assert mock_line.call_args[0][3] == tuple(color)

        # asserts that cv2.circle() is called 17 times (once for each keypoint)
        assert mock_circle.call_count == 17
        assert mock_line.call_count == 15
