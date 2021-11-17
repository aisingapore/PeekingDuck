"""
Copyright 2021 AI Singapore

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import pytest

from peekingduck.pipeline.nodes.dabble.keypoints_to_3d_loc import Node


@pytest.fixture
def node():
    return Node(
        {
            "input": "keypoints",
            "output": "obj_3d_locs",
            "focal_length": 1.14,
            "torso_factor": 0.9,
        }
    )


@pytest.fixture
def body_full():
    return np.array(
        [
            [0.6041, 0.5281],
            [0.6092, 0.506],
            [0.6022, 0.5054],
            [0.6566, 0.4902],
            [0.6269, 0.4973],
            [0.7009, 0.5484],
            [0.7129, 0.5131],
            [0.6413, 0.6407],
            [0.6543, 0.6211],
            [0.5496, 0.719],
            [0.5778, 0.6794],
            [0.7755, 0.6981],
            [0.7984, 0.6989],
            [0.7013, 0.8376],
            [0.8056, 0.8347],
            [0.698, 1.0109],
            [0.8787, 0.9799],
        ]
    )


@pytest.fixture
def body_some():
    return np.array(
        [
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [0.3869, 0.4124],
            [0.2929, 0.4412],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [0.3733, 0.6718],
            [0.3331, 0.6706],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-0.1, -1.0],
            [-1.0, -1.0],
        ]
    )


@pytest.fixture
def body_one():
    return np.array(
        [
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [0.2187, 0.999],
        ]
    )


@pytest.fixture
def torso_full():
    return np.array(
        [
            [0.6041, 0.5281],
            [0.7009, 0.5484],
            [0.7129, 0.5131],
            [0.7755, 0.6981],
            [0.7984, 0.6989],
        ]
    )


@pytest.fixture
def torso_one():
    return np.array([[0.2187, 0.999]])


@pytest.fixture
def bbox():
    return np.array([0.6041, 0.5281, 0.7009, 0.5484])


@pytest.fixture
def focal_length():
    return 1.14


@pytest.fixture
def torso_factor():
    return 0.9


class TestKeypointsTo3dLoc:
    def test_no_keypoints(self, node):
        array1 = []
        input1 = {"keypoints": array1}

        assert node.run(input1)["obj_3D_locs"] == []
        np.testing.assert_equal(input1["keypoints"], array1)

    def test_multi_keypoints(self, node, body_full, body_some):
        array1 = [body_full, body_some]
        input1 = {"keypoints": array1}

        assert len(node.run(input1)["obj_3D_locs"]) == 2
        np.testing.assert_equal(input1["keypoints"], array1)

    def test_one_keypoint(self, node, body_one):
        # scenario with not enough keypoints to form a bbox
        array1 = [body_one]
        input1 = {"keypoints": array1}

        assert len(node.run(input1)["obj_3D_locs"]) == 1
        np.testing.assert_equal(input1["keypoints"], array1)

    def test_get_torso_keypoints(self, node, body_full, body_some):
        ans_body_full = np.array(
            [
                [0.6041, 0.5281],
                [0.7009, 0.5484],
                [0.7129, 0.5131],
                [0.7755, 0.6981],
                [0.7984, 0.6989],
            ]
        )
        ans_body_some = np.array(
            [[0.3869, 0.4124], [0.2929, 0.4412], [0.3733, 0.6718], [0.3331, 0.6706]]
        )

        np.testing.assert_equal(ans_body_full, node._get_torso_keypoints(body_full))
        np.testing.assert_equal(ans_body_some, node._get_torso_keypoints(body_some))

    def test_enough_torso_keypoints(self, node, torso_full, torso_one):

        assert node._enough_torso_keypoints(torso_full)
        assert not node._enough_torso_keypoints(torso_one)

    def test_get_bbox(self, node, torso_full):
        ans_bbox = np.array([0.6041, 0.5131, 0.7984, 0.6989])

        np.testing.assert_equal(node._get_bbox(torso_full), ans_bbox)

    def test_get_3d_point_from_bbox(self, node, bbox, focal_length, torso_factor):
        ans_3d_point = np.array([6.7610, 1.6958, 50.5418])

        np.testing.assert_almost_equal(
            node._get_3d_point_from_bbox(bbox, focal_length, torso_factor),
            ans_3d_point,
            3,
        )
