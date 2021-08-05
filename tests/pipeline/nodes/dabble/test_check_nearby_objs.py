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
import pytest
import numpy as np
from peekingduck.pipeline.nodes.dabble.check_nearby_objs import Node

TAG_MSG = "TOO CLOSE!"


@pytest.fixture
def check_nearby_objs():
    node = Node({"input": ["obj_3D_locs"],
                 "output": ["obj_tags"],
                 "near_threshold": 2.0,
                 "tag_msg": TAG_MSG
                 })
    return node


class TestCheckNearbyObjs:
    def test_no_3D_locs(self, check_nearby_objs):
        array1 = []
        input1 = {"obj_3D_locs": array1}

        assert check_nearby_objs.run(input1)["obj_tags"] == []
        np.testing.assert_equal(input1["obj_3D_locs"], array1)

    def test_objs_are_nearby(self, check_nearby_objs):
        array1 = [np.array([0.5, 0.5, 0.5]),
                  np.array([0.55, 0.55, 0.55])]
        input1 = {"obj_3D_locs": array1}

        assert check_nearby_objs.run(input1)["obj_tags"] == [TAG_MSG, TAG_MSG]
        np.testing.assert_equal(input1["obj_3D_locs"], array1)

    def test_objs_not_nearby(self, check_nearby_objs):
        array1 = [np.array([0.1, 0.1, 0.1]),
                  np.array([0.9, 0.9, 3.0])]
        input1 = {"obj_3D_locs": array1}

        assert check_nearby_objs.run(input1)["obj_tags"] == ["", ""]
        np.testing.assert_equal(input1["obj_3D_locs"], array1)

    def test_some_nearby_some_not(self, check_nearby_objs):
        array1 = [np.array([0.1, 0.1, 0.1]),
                  np.array([0.1, 0.1, 6.0]),
                  np.array([0.1, 0.1, 1.0])]
        input1 = {"obj_3D_locs": array1}

        assert check_nearby_objs.run(input1)["obj_tags"] == [
            TAG_MSG, "", TAG_MSG]
        np.testing.assert_equal(input1["obj_3D_locs"], array1)
