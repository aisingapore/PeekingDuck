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
Test for draw tag node
"""

from contextlib import contextmanager
import numpy as np
import pytest

from peekingduck.pipeline.nodes.draw.tag import Node


@pytest.fixture
def tag_config():
    return {
        "input": ["bboxes", "obj_attrs", "img"],
        "output": ["none"],
        "show": [],
        "tag_color": [128, 128, 128],
    }


@pytest.fixture(
    params=[["one->two->three"], ["one -> two -> three"], ["  one->two  -> three  "]]
)
def nested_attr(request):
    yield request.param


class TestTag:
    def test_tag_drawn_on_img(self, tag_config, create_image):
        original_img = create_image((400, 400, 3))
        output_img = original_img.copy()
        input1 = {
            "img": output_img,
            "bboxes": [np.array([0, 0.5, 1, 1])],
            "obj_attrs": {"flags": ["TOO CLOSE!"]},
        }
        tag_config["show"] = ["flags"]
        Node(tag_config).run(input1)

        assert original_img.shape == output_img.shape
        np.testing.assert_raises(
            AssertionError, np.testing.assert_equal, original_img, output_img
        )

    def test_no_bboxes(self, tag_config, create_image):
        original_img = create_image((400, 400, 3))
        output_img = original_img.copy()
        input1 = {
            "img": output_img,
            "bboxes": [],
            "obj_attrs": {"flags": []},
        }
        tag_config["show"] = ["flags"]
        Node(tag_config).run(input1)
        np.testing.assert_equal(original_img, output_img)

    def test_empty_show_config(self, tag_config):
        tag_config["show"] = []
        with pytest.raises(KeyError) as excinfo:
            Node(tag_config)
        assert "config is currently empty" in str(excinfo.value)

    def test_str_tag(self, tag_config):
        tag_config["show"] = ["str"]
        draw_tag = Node(tag_config)
        input1 = {"str": ["a", "b", "c"]}
        tags = draw_tag._tags_from_obj_attrs(input1)
        assert tags == ["a", "b", "c"]

    def test_int_tag(self, tag_config):
        tag_config["show"] = ["int"]
        draw_tag = Node(tag_config)
        input1 = {"int": [1, 2, 3]}
        tags = draw_tag._tags_from_obj_attrs(input1)
        assert tags == ["1", "2", "3"]

    def test_float_tag(self, tag_config):
        tag_config["show"] = ["float"]
        draw_tag = Node(tag_config)
        input1 = {"float": [1.11, 2.22, 3.33]}
        tags = draw_tag._tags_from_obj_attrs(input1)
        assert tags == ["1.11", "2.22", "3.33"]

    def test_bool_tag(self, tag_config):
        tag_config["show"] = ["bool"]
        draw_tag = Node(tag_config)
        input1 = {"bool": [True, False, True]}
        tags = draw_tag._tags_from_obj_attrs(input1)
        assert tags == ["True", "False", "True"]

    def test_nested_attr(self, tag_config, nested_attr):
        tag_config["show"] = nested_attr
        draw_tag = Node(tag_config)
        input1 = {"one": {"two": {"three": ["a", "b", "c"]}}}
        tags = draw_tag._tags_from_obj_attrs(input1)
        assert tags == ["a", "b", "c"]

    def test_multiple_attr(self, tag_config):
        tag_config["show"] = ["one->two->three", "four", "five->six"]
        draw_tag = Node(tag_config)
        input1 = {
            "one": {"two": {"three": ["a", "b", "c"]}},
            "four": [1, 2, 3],
            "five": {"six": [1.11, 2.22, 3.33]},
        }
        tags = draw_tag._tags_from_obj_attrs(input1)
        assert tags == ["a, 1, 1.11", "b, 2, 2.22", "c, 3, 3.33"]

    def test_incorrect_attr_type(self, tag_config):
        tag_config["show"] = ["incorrect_type"]
        draw_tag = Node(tag_config)
        input1 = {"obj_attrs": {"incorrect_type": {"a": 8}}}
        with pytest.raises(TypeError) as excinfo:
            draw_tag.run(input1)
        assert "The attribute of interest has to be of type" in str(excinfo.value)

    def test_incorrect_tag_type(self, tag_config, create_image):
        original_img = create_image((400, 400, 3))
        output_img = original_img.copy()
        input1 = {
            "img": output_img,
            "bboxes": [np.array([0, 0.5, 1, 1])],
            "obj_attrs": {"type1": [1, 2, 3], "type2": [[1], [2], [3]]},
        }
        tag_config["show"] = ["type1", "type2"]
        draw_tag = Node(tag_config)
        with pytest.raises(TypeError) as excinfo:
            draw_tag.run(input1)
        assert "A tag has to be of type" in str(excinfo.value)

    def test_incorrect_color_format(self, tag_config):
        tag_config["tag_color"] = [128, -1, 128]
        with pytest.raises(ValueError) as excinfo:
            Node(tag_config)
        assert "Color values should lie between (and include) 0 and 255" in str(
            excinfo.value
        )

        tag_config["tag_color"] = [128, 256, 128]
        with pytest.raises(ValueError) as excinfo:
            Node(tag_config)
        assert "Color values should lie between (and include) 0 and 255" in str(
            excinfo.value
        )

        tag_config["tag_color"] = [128, 128.0, 128]
        with pytest.raises(TypeError) as excinfo:
            Node(tag_config)
        assert "Color values should be integers" in str(excinfo.value)

    def test_show_config_unchanged_between_frames(self, tag_config, create_image):
        # Each attr_key of the "show" config is obtained by recursion, where items in the
        # list are popped during recursion. deepcopy() is used to prevent "show" config
        # from being modified between frames and throwing an error that may be hard to trace.
        # This test is here to prevent the deepcopy() from being removed in future by accident.

        original_img = create_image((400, 400, 3))
        output_img = original_img.copy()
        tag_config["show"] = ["flags"]
        node = Node(tag_config)

        input1 = {
            "img": output_img,
            "bboxes": [np.array([0, 0.5, 1, 1])],
            "obj_attrs": {"flags": ["TOO CLOSE!"]},
        }
        node.run(input1)
        # This second run below should not throw an error if "show" config is unchanged
        with not_raises(TypeError):
            node.run(input1)


@contextmanager
def not_raises(exception):
    try:
        yield
    except exception:
        raise pytest.fail(f"DID RAISE EXCEPTION: {exception}")
