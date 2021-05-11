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
from peekingduck.pipeline.nodes.heuristic.bbox_count import Node


def create_node():
    node = Node({"input": ["bboxes"],
                 "output": ["count"],
                 })
    return node


class TestBboxCount:
    def test_no_bboxes(self):
        input1 = {"bboxes": []}
        node = create_node()
        assert node.run(input1)["count"] == 0

    def test_multi_bboxes(self):
        input1 = {"bboxes": [np.array([0.1, 0.2, 0.3, 0.4]),
                             np.array([0.5, 0.6, 0.7, 0.8])]}
        node = create_node()
        assert node.run(input1)["count"] == 2
