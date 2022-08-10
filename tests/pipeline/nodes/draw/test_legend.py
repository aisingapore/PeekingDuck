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

"""
Test for draw legend node
"""

import numpy as np
import pytest

from peekingduck.pipeline.nodes.draw.legend import Node


@pytest.fixture
def draw_legend_bottom():
    node = Node(
        {
            "input": ["all"],
            "output": ["img"],
            "show": ["fps", "count", "zone_count"],
            "position": "bottom",
            "box_opacity": 0.3,
            "font": {"size": 0.7, "thickness": 2},
        }
    )
    return node


@pytest.fixture
def draw_legend_top():
    node = Node(
        {
            "input": ["all"],
            "output": ["img"],
            "show": ["fps", "count", "zone_count"],
            "position": "top",
            "box_opacity": 0.3,
            "font": {"size": 0.7, "thickness": 2},
        }
    )
    return node


class TestLegend:
    def test_no_show_selected(self):
        with pytest.raises(KeyError) as excinfo:
            Node(
                {
                    "input": ["all"],
                    "output": ["img"],
                    "show": [],
                    "position": "bottom",
                    "box_opacity": 0.3,
                    "font": {"size": 0.7, "thickness": 2},
                }
            )
        assert (
            "To display information in the legend box, at least one data type must be selected"
            in str(excinfo.value)
        )

    def test_draw_legend_bottom_and_top(
        self, draw_legend_bottom, draw_legend_top, create_image
    ):
        original_img = create_image((640, 480, 3))
        output_img = original_img.copy()
        input1 = {"img": output_img, "fps": 50.5, "count": 2, "zone_count": [1, 1]}
        results_btm = draw_legend_bottom.run(input1)

        assert original_img.shape == results_btm["img"].shape
        np.testing.assert_raises(
            AssertionError, np.testing.assert_equal, original_img, results_btm["img"]
        )

        results_top = draw_legend_top.run(input1)
        assert original_img.shape == results_top["img"].shape
        np.testing.assert_raises(
            AssertionError, np.testing.assert_equal, original_img, results_top["img"]
        )

    def test_selected_data_type_not_in_data_pool(self, draw_legend_top, create_image):
        original_img = create_image((640, 480, 3))
        output_img = original_img.copy()
        input1 = {
            "img": output_img,
            "obj_groups": [4, 5],
        }
        with pytest.raises(KeyError) as excinfo:
            draw_legend_top.run(input1)
        assert (
            "was selected for drawing, but is not a valid data type from preceding nodes"
            in str(excinfo.value)
        )

    def test_invalid_draw_type(self, draw_legend_top, create_image):
        original_img = create_image((640, 480, 3))
        output_img = original_img.copy()
        input1 = {"img": output_img, "fps": [1, 1], "count": 2, "zone_count": [1, 1]}
        with pytest.raises(TypeError) as excinfo:
            draw_legend_top.run(input1)
        assert "the draw.legend node only draws values that are of type" in str(
            excinfo.value
        )
