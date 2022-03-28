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

import numpy as np
import pytest

from peekingduck.pipeline.nodes.dabble.group_nearby_objs import Node


@pytest.fixture
def group_nearby_objs():
    node = Node(
        {"input": ["obj_3D_locs"], "output": ["obj_attrs"], "obj_dist_threshold": 1.5}
    )
    return node


class TestGroupNearbyObjs:
    def test_no_3D_locs(self, group_nearby_objs):
        array1 = []
        input1 = {"obj_3D_locs": array1}

        assert group_nearby_objs.run(input1)["obj_attrs"]["groups"] == []
        np.testing.assert_equal(input1["obj_3D_locs"], array1)

    def test_objs_are_nearby(self, group_nearby_objs):
        array1 = [np.array([0.5, 0.5, 0.5]), np.array([0.55, 0.55, 0.55])]
        input1 = {"obj_3D_locs": array1}
        output1 = group_nearby_objs.run(input1)

        assert output1["obj_attrs"]["groups"][0] == output1["obj_attrs"]["groups"][1]
        assert input1["obj_3D_locs"] == array1
        np.testing.assert_equal(input1["obj_3D_locs"], array1)

    def test_objs_not_nearby(self, group_nearby_objs):
        array1 = [
            np.array([0.1, 0.1, 0.1]),
            np.array([0.1, 0.1, 3.0]),
            np.array([0.1, 0.1, 6.0]),
        ]
        input1 = {"obj_3D_locs": array1}
        # all different groups, should get [0, 1, 2]
        output1 = group_nearby_objs.run(input1)

        assert output1["obj_attrs"]["groups"][0] != output1["obj_attrs"]["groups"][1]
        assert output1["obj_attrs"]["groups"][1] != output1["obj_attrs"]["groups"][2]
        assert output1["obj_attrs"]["groups"][2] != output1["obj_attrs"]["groups"][0]
        np.testing.assert_equal(input1["obj_3D_locs"], array1)

    def test_multi_separate_groups(self, group_nearby_objs):
        array1 = [
            np.array([0.1, 0.1, 0.1]),
            np.array([0.1, 0.1, 6.0]),
            np.array([0.1, 0.1, 1.0]),
            np.array([0.1, 0.1, 5.0]),
        ]
        input1 = {"obj_3D_locs": array1}
        # should get something like [0, 1, 0, 1]
        output1 = group_nearby_objs.run(input1)

        assert output1["obj_attrs"]["groups"][0] != output1["obj_attrs"]["groups"][1]
        assert output1["obj_attrs"]["groups"][0] == output1["obj_attrs"]["groups"][2]
        assert output1["obj_attrs"]["groups"][1] == output1["obj_attrs"]["groups"][3]
        np.testing.assert_equal(input1["obj_3D_locs"], array1)
