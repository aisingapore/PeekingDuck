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
from peekingduck.pipeline.nodes.dabble.tag_count_over_time import Node


class TestTagCountOverTime:
    def test_no_tags(self):
        node = Node({"input": ["obj_tags"], "output": ["count"]})
        input1 = {"obj_tags": []}
        node.run(input1)
        input2 = {"obj_tags": []}
        assert node.run(input2)["count"] == 0

    def test_increasing_tag_num(self):
        node = Node({"input": ["obj_tags"], "output": ["count"]})
        input1 = {"obj_tags": ["0", "1", "2", "3", "4", "5", "6"]}
        node.run(input1)
        input2 = {"obj_tags": ["5", "6", "7"]}
        assert node.run(input2)["count"] == 8

    def test_decreasing_tag_num(self):
        # Scenario where no new objects to be tracked are introduced,
        # and some earlier tracked objects have disappeared from frame
        node = Node({"input": ["obj_tags"], "output": ["count"]})
        input1 = {"obj_tags": ["0", "1", "2", "3", "4", "5", "6"]}
        node.run(input1)
        input2 = {"obj_tags": ["0", "1", "2", "5"]}
        assert node.run(input2)["count"] == 7

    def test_mixed_tag_order(self):
        node = Node({"input": ["obj_tags"], "output": ["count"]})
        input1 = {"obj_tags": ["0", "1", "2", "3", "4", "5", "6"]}
        node.run(input1)
        input2 = {"obj_tags": ["5", "2", "6", "0", "7", "1", "4", "3"]}
        node.run(input2)
        input3 = {"obj_tags": ["5", "2", "3"]}
        assert node.run(input3)["count"] == 8

    def test_alternating_empty_nonempty(self):
        node = Node({"input": ["obj_tags"], "output": ["count"]})
        input1 = {"obj_tags": []}
        node.run(input1)
        input2 = {"obj_tags": ["0"]}
        node.run(input2)
        input3 = {"obj_tags": []}
        node.run(input3)
        input4 = {"obj_tags": ["0"]}
        assert node.run(input4)["count"] == 1
