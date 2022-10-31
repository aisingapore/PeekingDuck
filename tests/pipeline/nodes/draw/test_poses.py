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

import numpy as np
import pytest

from peekingduck.pipeline.nodes.draw.poses import Node


@pytest.fixture
def draw_poses():
    node = Node(
        {
            "input": ["keypoints", "keypoint_scores", "keypoint_conns", "img"],
            "output": ["none"],
            "keypoint_dot_color": [0, 255, 0],
            "keypoint_dot_radius": 5,
            "keypoint_connect_color": [0, 255, 255],
            "keypoint_text_color": [255, 0, 255],
        }
    )
    return node


class TestPoses:
    def test_no_poses(self, draw_poses, create_image):
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
        draw_poses.run(inputs)

        np.testing.assert_equal(original_img, output_img)
