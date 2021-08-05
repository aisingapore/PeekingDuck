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
from peekingduck.pipeline.nodes.dabble.check_large_groups import Node


@pytest.fixture
def check_large_groups():
    node = Node({"input": ["obj_groups"],
                 "output": ["large_groups"],
                 "group_size_thres": 3
                 })
    return node


class TestCheckLargeGroups:
    def test_no_obj_groups(self, check_large_groups):
        array1 = []
        input1 = {"obj_groups": array1}

        assert check_large_groups.run(input1)["large_groups"] == []
        assert input1["obj_groups"] == array1

    def test_no_large_groups(self, check_large_groups):
        array1 = [0, 1, 2, 3, 4, 5]
        input1 = {"obj_groups": array1}

        assert check_large_groups.run(input1)["large_groups"] == []
        assert input1["obj_groups"] == array1

    def test_multi_large_groups(self, check_large_groups):
        array1 = [0, 1, 0, 3, 1, 0, 1, 2, 1, 0]
        input1 = {"obj_groups": array1}

        assert check_large_groups.run(input1)["large_groups"] == [0, 1]
        assert input1["obj_groups"] == array1
