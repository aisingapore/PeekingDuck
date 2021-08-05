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
from peekingduck.pipeline.nodes.dabble.bbox_to_3d_loc import Node


@pytest.fixture
def bbox_to_3d_loc():
    node = Node({"input": "bboxes",
                 "output": "obj_3d_locs",
                 "focal_length": 1.14,
                 "height_factor": 2.5
                 })
    return node


class TestBboxTo3dLoc:
    def test_no_bbox(self, bbox_to_3d_loc):
        array1 = []
        input1 = {"bboxes": array1}

        assert bbox_to_3d_loc.run(input1)["obj_3D_locs"] == []
        np.testing.assert_equal(input1["bboxes"], array1)

    def test_multi_bboxes(self, bbox_to_3d_loc):
        array1 = [np.array([0.1, 0.2, 0.3, 0.4]),
                  np.array([0.5, 0.6, 0.7, 0.8])]
        input1 = {"bboxes": array1}

        assert len(bbox_to_3d_loc.run(input1)["obj_3D_locs"]) == 2
        np.testing.assert_equal(input1["bboxes"], array1)

    def test_formula(self, bbox_to_3d_loc):
        array1 = [np.array([0.408, 0.277, 0.894, 1.0])]
        input1 = {"bboxes": array1}
        output1 = bbox_to_3d_loc.run(input1)["obj_3D_locs"]
        correct_ans = [np.array([0.522, 0.479, 3.942])]

        np.testing.assert_almost_equal(output1, correct_ans, decimal=3)
        np.testing.assert_equal(input1["bboxes"], array1)
