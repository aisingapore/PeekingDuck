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

import os
import pytest
import numpy as np
from peekingduck.pipeline.nodes.dabble.tracking import Node


@pytest.fixture
def size():
    return (400, 600, 3)

@pytest.fixture
def track_config():
    node_config = dict()
    node_config['root'] = os.getcwd()
    node_config["input"] = ["img", "bboxes", "bbox_scores", "bbox_labels"]
    node_config["output"] = ["obj_tags"]
    return node_config

@pytest.fixture(params=['iou', 'mosse'])
def tracker(request, track_config):
    track_config['tracking_type'] = request.param
    node = Node(track_config)
    return node


class TestTracking:
    def test_no_tags(self, create_image, size, tracker):
        img1 = create_image(size)
        array1 = []
        array2 = []
        array3 = []
        input1 = {"img": img1, "bboxes": array1,
                  "bbox_scores": array2, "bbox_labels": array3}

        assert tracker.run(input1)["obj_tags"] == []
        np.testing.assert_equal(input1["img"], img1)
        np.testing.assert_equal(input1["bboxes"], array1)

    def test_multi_tags(self, create_image, size, tracker):
        img1 = create_image(size)
        array1 = [np.array([0.1, 0.2, 0.3, 0.4]),
                  np.array([0.5, 0.6, 0.7, 0.8])]
        array2 = [0.9, 0.6]
        array3 = ["label1", "label2"]
        input1 = {"img": img1, "bboxes": array1,
                  "bbox_scores": array2, "bbox_labels": array3}

        assert len(tracker.run(input1)["obj_tags"]) == 2
        assert len(tracker.run(input1)["obj_tags"]) == len(array1)
        np.testing.assert_equal(input1["img"], img1)
        np.testing.assert_equal(input1["bboxes"], array1)
