# Copyright 2021 AI Singapore
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
Test for draw legend node
"""
import pytest
import numpy as np
from peekingduck.pipeline.nodes.draw.legend import Node

@pytest.fixture
def draw_legend_bottom():
    node = Node({'input': ["all"],
                 'output': ["img"],
                 'all_legend_items': ["fps", "count", "zone_count"],
                 'position': "bottom",
                 'include': ["all_legend_items"]
                })
    return node

@pytest.fixture
def draw_legend_top():
    node = Node({'input': ["all"],
                 'output': ["img"],
                 'all_legend_items': ["fps", "count", "zone_count"],
                 'position': "top",
                 'include': ["all_legend_items"]
                })
    return node

@pytest.fixture
def draw_legend_fps_only():
    node = Node({'input': ["all"],
                 'output': ["img"],
                 'all_legend_items': ["fps", "count", "zone_count"],
                 'position': "top",
                 'include': ["fps"]
                })
    return node


class TestLegend:
    def test_no_relevant_inputs(self, draw_legend_bottom, create_image):
        original_img = create_image((28, 28, 3))
        input1 = {
        "img": original_img
        }
        expected_output = {}
        results = draw_legend_bottom.run(input1)
        assert results == expected_output

    # formula: processed image = contrast * image + brightness
    def test_draw_legend_botom_and_top(self, draw_legend_bottom, 
                                       draw_legend_top, create_image):
        original_img = create_image((640, 480, 3))
        output_img = original_img.copy()
        input1 = {
        "img": output_img,
        "fps": 50.5,
        "count": 2,
        "zone_count": [1, 1]
        }
        results_btm = draw_legend_bottom.run(input1)

        assert results_btm != {}
        assert original_img.shape == results_btm['img'].shape
        np.testing.assert_raises(AssertionError, np.testing.assert_equal,
                                 original_img, results_btm['img'])

        results_top = draw_legend_top.run(input1)
        np.testing.assert_raises(AssertionError, np.testing.assert_equal,
                                 original_img, results_top)

    def test_draw_fps_only(self, draw_legend_fps_only, create_image):
        original_img = create_image((640, 480, 3))
        output_img = original_img.copy()
        input1 = {
        "img": output_img,
        "fps": 50.5,
        }
        results = draw_legend_fps_only.run(input1)

        np.testing.assert_raises(AssertionError, np.testing.assert_equal,
                                 original_img, results['img'])
