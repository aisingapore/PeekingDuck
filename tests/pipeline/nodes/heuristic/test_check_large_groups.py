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
from peekingduck.pipeline.nodes.heuristic.check_large_groups import Node


def create_node():
    node = Node({"input": ["obj_groups"],
                 "output": ["large_groups"],
                 "group_size_thres": 3
                 })
    return node


class TestCheckLargeGroups:
    def test_no_obj_groups(self):
        input1 = {"obj_groups": []}
        node = create_node()
        assert node.run(input1)["large_groups"] == []

    def test_no_large_groups(self):
        input1 = {"obj_groups": [0, 1, 2, 3, 4, 5]}
        node = create_node()
        assert node.run(input1)["large_groups"] == []

    def test_multi_large_groups(self):
        input1 = {"obj_groups": [0, 1, 0, 3, 1, 0, 1, 2, 1, 0]}
        node = create_node()
        assert node.run(input1)["large_groups"] == [0, 1]
