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

import pytest
import yaml

from peekingduck.pipeline.nodes.base import ThresholdCheckerMixin
from tests.conftest import not_raises


@pytest.fixture
def threshold_model():
    return ThresholdModel()


class ThresholdModel(ThresholdCheckerMixin):
    def __init__(self):
        self.config = {"a": 12.5, "b": 10, "c": "value", "d": [10, 12.5, 13]}


class TestThresholdCheckerMixin:
    @pytest.mark.parametrize("include", ["lower", "both"])
    def test_threshold_above_value(self, threshold_model, include):
        with not_raises(ValueError):
            # single value
            threshold_model.check_bounds("a", 9, "above", include)
            # single value, inclusive
            threshold_model.check_bounds("a", 12, "above", include)
            # multiple values
            threshold_model.check_bounds(["a", "b", "d"], 9, "above", include)
            # multiple values, inclusive
            threshold_model.check_bounds(["a", "b", "d"], 10, "above", include)

        with pytest.raises(ValueError) as excinfo:
            # single value, fail
            threshold_model.check_bounds("a", 13, "above", include)
        assert "a must be more than or equal to 13" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail first
            threshold_model.check_bounds(["a", "b"], 13, "above", include)
        assert "a must be more than or equal to 13" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail second
            threshold_model.check_bounds(["a", "b"], 11, "above", include)
        assert "b must be more than or equal to 11" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail second
            threshold_model.check_bounds(["a", "d"], 11, "above", include)
        assert "All elements of d must be more than or equal to 11" == str(
            excinfo.value
        )

    @pytest.mark.parametrize("include", ["upper", None])
    def test_threshold_above_value_exclusive(self, threshold_model, include):
        with not_raises(ValueError):
            # single value
            threshold_model.check_bounds("a", 9, "above", include)
            # multiple values
            threshold_model.check_bounds(["a", "b"], 9, "above", include)

        with pytest.raises(ValueError) as excinfo:
            # single value, inclusive, fail
            threshold_model.check_bounds("a", 12.5, "above", include)
        assert "a must be more than 12.5" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, fail
            threshold_model.check_bounds("a", 13, "above", include)
        assert "a must be more than 13" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, fail
            threshold_model.check_bounds("d", 13, "above", include)
        assert "All elements of d must be more than 13" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, inclusive, fail first
            threshold_model.check_bounds(["a", "b"], 12.5, "above", include)
        assert "a must be more than 12.5" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, inclusive, fail second
            threshold_model.check_bounds(["a", "b"], 10, "above", include)
        assert "b must be more than 10" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail first
            threshold_model.check_bounds(["a", "b"], 13, "above", include)
        assert "a must be more than 13" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail second
            threshold_model.check_bounds(["a", "b"], 11, "above", include)
        assert "b must be more than 11" == str(excinfo.value)

    def test_threshold_above_value_invalid_type(self, threshold_model):
        with pytest.raises(TypeError) as excinfo:
            threshold_model.check_bounds({"key1": "a"}, 10, "above")
        assert "`key` must be either str or list" == str(excinfo.value)

    @pytest.mark.parametrize("include", ["upper", "both"])
    def test_threshold_below_value(self, threshold_model, include):
        with not_raises(ValueError):
            # single value
            threshold_model.check_bounds("a", 13, "below", include)
            # single value, inclusive
            threshold_model.check_bounds("a", 12.5, "below", include)
            # multiple values
            threshold_model.check_bounds(["b", "a"], 13, "below", include)
            # multiple values, inclusive
            threshold_model.check_bounds(["b", "a"], 12.5, "below", include)
            # multiple values, inclusive
            threshold_model.check_bounds(["b", "a", "d"], 13, "below", include)

        with pytest.raises(ValueError) as excinfo:
            # single value, fail
            threshold_model.check_bounds("a", 12, "below", include)
        assert "a must be less than or equal to 12" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail first
            threshold_model.check_bounds(["b", "a"], 9, "below", include)
        assert "b must be less than or equal to 9" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail second
            threshold_model.check_bounds(["b", "a"], 11, "below", include)
        assert "a must be less than or equal to 11" == str(excinfo.value)

    @pytest.mark.parametrize("include", ["lower", None])
    def test_threshold_below_value_exclusive(self, threshold_model, include):
        with not_raises(ValueError):
            # single value
            threshold_model.check_bounds("a", 13, "below", include)
            # multiple values
            threshold_model.check_bounds(["a", "b"], 13, "below", include)

        with pytest.raises(ValueError) as excinfo:
            # single value, inclusive, fail
            threshold_model.check_bounds("a", 12.5, "below", include)
        assert "a must be less than 12.5" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, inclusive, fail
            threshold_model.check_bounds("d", 13, "below", include)
        assert "All elements of d must be less than 13" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, fail
            threshold_model.check_bounds("a", 12, "below", include)
        assert "a must be less than 12" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, inclusive, fail first
            threshold_model.check_bounds(["b", "a"], 9, "below", include)
        assert "b must be less than 9" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, inclusive, fail second
            threshold_model.check_bounds(["b", "a"], 12, "below", include)
        assert "a must be less than 12" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail first
            threshold_model.check_bounds(["b", "a"], 9, "below", include)
        assert "b must be less than 9" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail second
            threshold_model.check_bounds(["b", "a"], 12, "below", include)
        assert "a must be less than 12" == str(excinfo.value)

    def test_threshold_below_value_invalid_type(self, threshold_model):
        with pytest.raises(TypeError) as excinfo:
            threshold_model.check_bounds({"key1": "a"}, 10, "below")
        assert "`key` must be either str or list" == str(excinfo.value)

    def test_threshold_within_bounds(self, threshold_model):
        with not_raises(ValueError):
            # single value
            threshold_model.check_bounds("a", (9, 13), "within")
            # single value, inclusive lower
            threshold_model.check_bounds("a", (12.5, 13), "within")
            # single value, inclusive upper
            threshold_model.check_bounds("a", (9, 12.5), "within")
            # multiple value
            threshold_model.check_bounds(["a", "b"], (9, 13), "within")
            # multiple value, inclusive lower
            threshold_model.check_bounds(["a", "b"], (10, 13), "within")
            # multiple value, inclusive upper
            threshold_model.check_bounds(["a", "b"], (9, 12.5), "within")
            # multiple value, inclusive both
            threshold_model.check_bounds(["a", "b"], (10, 12.5), "within")
            # multiple value, inclusive both
            threshold_model.check_bounds(["a", "b", "d"], (10, 13), "within")

        with pytest.raises(ValueError) as excinfo:
            # single value, fail lower
            threshold_model.check_bounds("a", (13, 14), "within")
        assert "a must be between [13, 14]" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, fail upper
            threshold_model.check_bounds("a", (9, 11), "within")
        assert "a must be between [9, 11]" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail lower
            threshold_model.check_bounds(["a", "b"], (11, 13), "within")
        assert "b must be between [11, 13]" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail upper
            threshold_model.check_bounds(["a", "b"], (9, 11), "within")
        assert "a must be between [9, 11]" == str(excinfo.value)

    def test_threshold_within_bounds_exclusive_lower(self, threshold_model):
        with not_raises(ValueError):
            # single value
            threshold_model.check_bounds("a", (9, 13), "within", include="upper")
            # single value, inclusive upper
            threshold_model.check_bounds("a", (9, 12.5), "within", include="upper")
            # multiple value
            threshold_model.check_bounds(["a", "b"], (9, 13), "within", include="upper")
            # multiple value, inclusive upper
            threshold_model.check_bounds(
                ["a", "b"], (9, 12.5), "within", include="upper"
            )

        with pytest.raises(ValueError) as excinfo:
            # single value, inclusive lower, fail
            threshold_model.check_bounds("a", (12.5, 13), "within", include="upper")
        assert "a must be between (12.5, 13]" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, inclusive lower, fail
            threshold_model.check_bounds("d", (10, 13), "within", include="upper")
        assert "All elements of d must be between (10, 13]" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, above bounds, fail
            threshold_model.check_bounds("a", (9, 11), "within", include="upper")
        assert "a must be between (9, 11]" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, below bounds, fail
            threshold_model.check_bounds("a", (13, 14), "within", include="upper")
        assert "a must be between (13, 14]" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, inclusive lower, fail
            threshold_model.check_bounds(
                ["a", "b"], (10, 13), "within", include="upper"
            )
        assert "b must be between (10, 13]" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, inclusive upper, fail
            threshold_model.check_bounds(
                ["a", "b"], (11, 12.5), "within", include="upper"
            )
        assert "b must be between (11, 12.5]" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, inclusive both
            threshold_model.check_bounds(
                ["a", "b"], (10, 12.5), "within", include="upper"
            )
        assert "b must be between (10, 12.5]" == str(excinfo.value)

    def test_threshold_within_bounds_exclusive_upper(self, threshold_model):
        with not_raises(ValueError):
            # single value
            threshold_model.check_bounds("a", (9, 13), "within", include="lower")
            # single value, inclusive lower
            threshold_model.check_bounds("a", (12.5, 13), "within", include="lower")
            # multiple value
            threshold_model.check_bounds(["a", "b"], (9, 13), "within", include="lower")
            # multiple value, inclusive lower
            threshold_model.check_bounds(
                ["a", "b"], (10, 13), "within", include="lower"
            )

        with pytest.raises(ValueError) as excinfo:
            # single value, fail lower
            threshold_model.check_bounds("a", (13, 14), "within", include="lower")
        assert "a must be between [13, 14)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, fail upper
            threshold_model.check_bounds("a", (9, 11), "within", include="lower")
        assert "a must be between [9, 11)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, inclusive upper, fail
            threshold_model.check_bounds("a", (9, 12.5), "within", include="lower")
        assert "a must be between [9, 12.5)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, inclusive upper, fail
            threshold_model.check_bounds("d", (9, 13), "within", include="lower")
        assert "All elements of d must be between [9, 13)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail lower
            threshold_model.check_bounds(
                ["a", "b"], (11, 13), "within", include="lower"
            )
        assert "b must be between [11, 13)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail upper
            threshold_model.check_bounds(["a", "b"], (9, 11), "within", include="lower")
        assert "a must be between [9, 11)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, inclusive upper, fail
            threshold_model.check_bounds(
                ["a", "b"], (9, 12.5), "within", include="lower"
            )
        assert "a must be between [9, 12.5)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, inclusive both, fail
            threshold_model.check_bounds(
                ["a", "b"], (10, 12.5), "within", include="lower"
            )
        assert "a must be between [10, 12.5)" == str(excinfo.value)

    def test_threshold_within_bounds_exclusive_both(self, threshold_model):
        with not_raises(ValueError):
            # single value
            threshold_model.check_bounds("a", (9, 13), "within", include=None)
            # multiple value
            threshold_model.check_bounds(["a", "b"], (9, 13), "within", include=None)
            # multiple value
            threshold_model.check_bounds(
                ["a", "b", "d"], (9, 13.5), "within", include=None
            )

        with pytest.raises(ValueError) as excinfo:
            # single value, fail lower
            threshold_model.check_bounds("a", (13, 14), "within", include=None)
        assert "a must be between (13, 14)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, fail upper
            threshold_model.check_bounds("a", (9, 11), "within", include=None)
        assert "a must be between (9, 11)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, inclusive lower, fail
            threshold_model.check_bounds("a", (12.5, 13), "within", include=None)
        assert "a must be between (12.5, 13)" == str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            # single value, inclusive upper, fail
            threshold_model.check_bounds("a", (9, 12.5), "within", include=None)
        assert "a must be between (9, 12.5)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail lower
            threshold_model.check_bounds(["a", "b"], (11, 13), "within", include=None)
        assert "b must be between (11, 13)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail upper
            threshold_model.check_bounds(["a", "b"], (9, 11), "within", include=None)
        assert "a must be between (9, 11)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, inclusive lower, fail
            threshold_model.check_bounds(["a", "b"], (10, 13), "within", include=None)
        assert "b must be between (10, 13)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, inclusive upper, fail
            threshold_model.check_bounds(["a", "b"], (9, 12.5), "within", include=None)
        assert "a must be between (9, 12.5)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, inclusive both, fail
            threshold_model.check_bounds(["a", "b"], (10, 12.5), "within", include=None)
        # Implementation quirk: checks all lower bounds first and then upper
        # bound instead of checking both bounds following the key order
        assert "b must be between (10, 12.5)" == str(excinfo.value)

    def test_threshold_within_bounds_invalid_type(self, threshold_model):
        with pytest.raises(TypeError) as excinfo:
            threshold_model.check_bounds({"key1": "a"}, (10, 12), "within")
        assert "`key` must be either str or list" == str(excinfo.value)

    def test_threshold_invalid_method(self, threshold_model):
        with pytest.raises(ValueError) as excinfo:
            threshold_model.check_bounds("a", 12, "invalid_method")
        assert "`method` must be one of" in str(excinfo.value)

    def test_threshold_invalid_tuple_type(self, threshold_model):
        with pytest.raises(TypeError) as excinfo:
            threshold_model.check_bounds("a", (12, "a"), "within")
        assert "When using tuple for `value`, it must be a tuple of float/int" == str(
            excinfo.value
        )

    def test_threshold_invalid_tuple_length(self, threshold_model):
        with pytest.raises(ValueError) as excinfo:
            threshold_model.check_bounds("a", (1, 2, 3), "within")
        assert "When using tuple for `value`, it must contain only 2 elements" == str(
            excinfo.value
        )

    def test_threshold_invalid_value_type(self, threshold_model):
        value = {1, 2}
        with pytest.raises(TypeError) as excinfo:
            threshold_model.check_bounds("a", value, "within")
        assert (
            f"`value` must be a float/int or tuple, but you passed a {type(value).__name__}"
            == str(excinfo.value)
        )

    def test_threshold_invalid_within_value_type(self, threshold_model):
        with pytest.raises(TypeError) as excinfo:
            threshold_model.check_bounds("a", 1, "within")
        assert "`value` must be a tuple when `method` is 'within'" == str(excinfo.value)

    @pytest.mark.parametrize("method", ["above", "below"])
    def test_threshold_invalid_above_below_value_type(self, threshold_model, method):
        with pytest.raises(TypeError) as excinfo:
            threshold_model.check_bounds("a", (1, 2), method)
        assert "`value` must be a float/int when `method` is 'above'/'below'" == str(
            excinfo.value
        )

    def test_threshold_valid_choice(self, threshold_model):
        with not_raises(ValueError):
            threshold_model.check_valid_choice("a", {"value", 10, 11, 12.5})
            threshold_model.check_valid_choice("b", {"value", 10, 11, 12.5})
            threshold_model.check_valid_choice("c", {"value", 10, 11, 12.5})

        with pytest.raises(ValueError) as excinfo:
            threshold_model.check_valid_choice("a", {9, 10, 11})
        assert "a must be one of {9, 10, 11}" == str(excinfo.value)

    def test_threshold_valid_choice_invalid_type(self, threshold_model):
        with pytest.raises(TypeError) as excinfo:
            threshold_model.check_valid_choice({"key": "a"}, {9, 10, 11})
        assert "`key` must be str" == str(excinfo.value)
