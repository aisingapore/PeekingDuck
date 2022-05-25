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

from peekingduck.pipeline.nodes.base import ThresholdCheckerMixin
from tests.conftest import not_raises


@pytest.fixture
def threshold_model():
    return ThresholdModel()


class ThresholdModel(ThresholdCheckerMixin):
    def __init__(self):
        self.config = {"a": 12.5, "b": 10, "c": "value", "d": [10, 12.5, 13]}


class TestThresholdCheckerMixin:
    def test_check_bounds_invalid_type(self, threshold_model):
        with pytest.raises(TypeError) as excinfo:
            threshold_model.check_bounds({"key1": "a"}, "[10, +inf]")
        assert "`key` must be either str or list" == str(excinfo.value)

    def test_check_bounds_bad_format(self, threshold_model):
        with pytest.raises(ValueError) as excinfo:
            # missing separator
            threshold_model.check_bounds("a", "[13 14]")
        assert "Badly formatted interval" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # wrong separator
            threshold_model.check_bounds("a", "[13; 14]")
        assert "Badly formatted interval" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # missing left bracket
            threshold_model.check_bounds("a", "13, 14]")
        assert "Badly formatted interval" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # missing right bracket
            threshold_model.check_bounds("a", "[13, 14")
        assert "Badly formatted interval" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # wrong left bracket
            threshold_model.check_bounds("a", "{13, 14]")
        assert "Badly formatted interval" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # wrong right bracket
            threshold_model.check_bounds("a", "[13, 14}")
        assert "Badly formatted interval" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # wrong right bracket
            threshold_model.check_bounds("a", "[13, 14}")
        assert "Badly formatted interval" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # wrong left value
            threshold_model.check_bounds("a", "[value, 14]")
        assert "Badly formatted interval" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # wrong left value
            threshold_model.check_bounds("a", "[1 2, 14]")
        assert "Badly formatted interval" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # wrong right value
            threshold_model.check_bounds("a", "[13, 14.5.6]")
        assert "Badly formatted interval" == str(excinfo.value)

    def test_check_bounds_invalid_interval(self, threshold_model):
        with pytest.raises(ValueError) as excinfo:
            threshold_model.check_bounds("a", "[14, 13]")
        assert "Lower bound cannot be larger than upper bound" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            threshold_model.check_bounds("a", "[+inf, 13]")
        assert "Lower bound cannot be larger than upper bound" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            threshold_model.check_bounds("a", "[14, -inf]")
        assert "Lower bound cannot be larger than upper bound" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            threshold_model.check_bounds("a", "[+inf, -inf]")
        assert "Lower bound cannot be larger than upper bound" == str(excinfo.value)

    @pytest.mark.parametrize("include", [")", "]"])
    def test_threshold_above_value(self, threshold_model, include):
        with not_raises(ValueError):
            # single value
            threshold_model.check_bounds("a", f"[9, +inf{include}")
            # single value, inclusive
            threshold_model.check_bounds("a", f"[12, +inf{include}")
            # multiple values
            threshold_model.check_bounds(["a", "b", "d"], f"[9, +inf{include}")
            # multiple values, inclusive
            threshold_model.check_bounds(["a", "b", "d"], f"[10, +inf{include}")

        with pytest.raises(ValueError) as excinfo:
            # single value, fail
            threshold_model.check_bounds("a", f"[13, +inf{include}")
        assert f"a must be between [13.0, inf{include}" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail first
            threshold_model.check_bounds(["a", "b"], f"[13, +inf{include}")
        assert f"a must be between [13.0, inf{include}" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail second
            threshold_model.check_bounds(["a", "b"], f"[11, +inf{include}")
        assert f"b must be between [11.0, inf{include}" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail second
            threshold_model.check_bounds(["a", "d"], f"[11, +inf{include}")
        assert f"All elements of d must be between [11.0, inf{include}" == str(
            excinfo.value
        )

    @pytest.mark.parametrize("include", [")", "]"])
    def test_threshold_above_value_exclusive(self, threshold_model, include):
        with not_raises(ValueError):
            # single value
            threshold_model.check_bounds("a", f"(9, +inf{include}")
            # multiple values
            threshold_model.check_bounds(["a", "b"], f"(9, +inf{include}")

        with pytest.raises(ValueError) as excinfo:
            # single value, inclusive, fail
            threshold_model.check_bounds("a", f"(12.5, +inf{include}")
        assert f"a must be between (12.5, inf{include}" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, fail
            threshold_model.check_bounds("a", f"(13, +inf{include}")
        assert f"a must be between (13.0, inf{include}" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, fail
            threshold_model.check_bounds("d", f"(13, +inf{include}")
        assert f"All elements of d must be between (13.0, inf{include}" == str(
            excinfo.value
        )

        with pytest.raises(ValueError) as excinfo:
            # multiple value, inclusive, fail first
            threshold_model.check_bounds(["a", "b"], f"(12.5, +inf{include}")
        assert f"a must be between (12.5, inf{include}" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, inclusive, fail second
            threshold_model.check_bounds(["a", "b"], f"(10, +inf{include}")
        assert f"b must be between (10.0, inf{include}" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail first
            threshold_model.check_bounds(["a", "b"], f"(13, +inf{include}")
        assert f"a must be between (13.0, inf{include}" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail second
            threshold_model.check_bounds(["a", "b"], f"(11, +inf{include}")
        assert f"b must be between (11.0, inf{include}" == str(excinfo.value)

    @pytest.mark.parametrize("include", ["(", "["])
    def test_threshold_below_value(self, threshold_model, include):
        with not_raises(ValueError):
            # single value
            threshold_model.check_bounds("a", f"{include}-inf, 13]")
            # single value, inclusive
            threshold_model.check_bounds("a", f"{include}-inf, 12.5]")
            # multiple values
            threshold_model.check_bounds(["b", "a"], f"{include}-inf, 13]")
            # multiple values, inclusive
            threshold_model.check_bounds(["b", "a"], f"{include}-inf, 12.5]")
            # multiple values, inclusive
            threshold_model.check_bounds(["b", "a", "d"], f"{include}-inf, 13]")

        with pytest.raises(ValueError) as excinfo:
            # single value, fail
            threshold_model.check_bounds("a", f"{include}-inf, 12]")
        assert f"a must be between {include}-inf, 12.0]" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail first
            threshold_model.check_bounds(["b", "a"], f"{include}-inf, 9]")
        assert f"b must be between {include}-inf, 9.0]" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail second
            threshold_model.check_bounds(["b", "a"], f"{include}-inf, 11]")
        assert f"a must be between {include}-inf, 11.0]" == str(excinfo.value)

    @pytest.mark.parametrize("include", ["(", "["])
    def test_threshold_below_value_exclusive(self, threshold_model, include):
        with not_raises(ValueError):
            # single value
            threshold_model.check_bounds("a", f"{include}-inf, 13)")
            # multiple values
            threshold_model.check_bounds(["a", "b"], f"{include}-inf, 13)")

        with pytest.raises(ValueError) as excinfo:
            # single value, inclusive, fail
            threshold_model.check_bounds("a", f"{include}-inf, 12.5)")
        assert f"a must be between {include}-inf, 12.5)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, inclusive, fail
            threshold_model.check_bounds("d", f"{include}-inf, 13)")
        assert f"All elements of d must be between {include}-inf, 13.0)" == str(
            excinfo.value
        )

        with pytest.raises(ValueError) as excinfo:
            # single value, fail
            threshold_model.check_bounds("a", f"{include}-inf, 12)")
        assert f"a must be between {include}-inf, 12.0)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, inclusive, fail first
            threshold_model.check_bounds(["b", "a"], f"{include}-inf, 9)")
        assert f"b must be between {include}-inf, 9.0)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, inclusive, fail second
            threshold_model.check_bounds(["b", "a"], f"{include}-inf, 12)")
        assert f"a must be between {include}-inf, 12.0)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail first
            threshold_model.check_bounds(["b", "a"], f"{include}-inf, 9)")
        assert f"b must be between {include}-inf, 9.0)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail second
            threshold_model.check_bounds(["b", "a"], f"{include}-inf, 12)")
        assert f"a must be between {include}-inf, 12.0)" == str(excinfo.value)

    def test_threshold_within_bounds(self, threshold_model):
        with not_raises(ValueError):
            # single value
            threshold_model.check_bounds("a", "[9, 13]")
            # single value, inclusive lower
            threshold_model.check_bounds("a", "[12.5, 13]")
            # single value, inclusive upper
            threshold_model.check_bounds("a", "[9, 12.5]")
            # multiple value
            threshold_model.check_bounds(["a", "b"], "[9, 13]")
            # multiple value, inclusive lower
            threshold_model.check_bounds(["a", "b"], "[10, 13]")
            # multiple value, inclusive upper
            threshold_model.check_bounds(["a", "b"], "[9, 12.5]")
            # multiple value, inclusive both
            threshold_model.check_bounds(["a", "b"], "[10, 12.5]")
            # multiple value, inclusive both
            threshold_model.check_bounds(["a", "b", "d"], "[10, 13]")

        with pytest.raises(ValueError) as excinfo:
            # single value, fail lower
            threshold_model.check_bounds("a", "[13, 14]")
        assert "a must be between [13.0, 14.0]" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, fail upper
            threshold_model.check_bounds("a", "[9, 11]")
        assert "a must be between [9.0, 11.0]" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail lower
            threshold_model.check_bounds(["a", "b"], "[11, 13]")
        assert "b must be between [11.0, 13.0]" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail upper
            threshold_model.check_bounds(["a", "b"], "[9, 11]")
        assert "a must be between [9.0, 11.0]" == str(excinfo.value)

    def test_threshold_within_bounds_exclusive_lower(self, threshold_model):
        with not_raises(ValueError):
            # single value
            threshold_model.check_bounds("a", "(9, 13]")
            # single value, inclusive upper
            threshold_model.check_bounds("a", "(9, 12.5]")
            # multiple value
            threshold_model.check_bounds(["a", "b"], "(9, 13]")
            # multiple value, inclusive upper
            threshold_model.check_bounds(["a", "b"], "(9, 12.5]")

        with pytest.raises(ValueError) as excinfo:
            # single value, inclusive lower, fail
            threshold_model.check_bounds("a", "(12.5, 13]")
        assert "a must be between (12.5, 13.0]" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, inclusive lower, fail
            threshold_model.check_bounds("d", "(10, 13]")
        assert "All elements of d must be between (10.0, 13.0]" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, above bounds, fail
            threshold_model.check_bounds("a", "(9, 11]")
        assert "a must be between (9.0, 11.0]" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, below bounds, fail
            threshold_model.check_bounds("a", "(13, 14]")
        assert "a must be between (13.0, 14.0]" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, inclusive lower, fail
            threshold_model.check_bounds(["a", "b"], "(10, 13]")
        assert "b must be between (10.0, 13.0]" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, inclusive upper, fail
            threshold_model.check_bounds(["a", "b"], "(11, 12.5]")
        assert "b must be between (11.0, 12.5]" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, inclusive both
            threshold_model.check_bounds(["a", "b"], "(10, 12.5]")
        assert "b must be between (10.0, 12.5]" == str(excinfo.value)

    def test_threshold_within_bounds_exclusive_upper(self, threshold_model):
        with not_raises(ValueError):
            # single value
            threshold_model.check_bounds("a", "[9, 13)")
            # single value, inclusive lower
            threshold_model.check_bounds("a", "[12.5, 13)")
            # multiple value
            threshold_model.check_bounds(["a", "b"], "[9, 13)")
            # multiple value, inclusive lower
            threshold_model.check_bounds(["a", "b"], "[10, 13)")

        with pytest.raises(ValueError) as excinfo:
            # single value, fail lower
            threshold_model.check_bounds("a", "[13, 14)")
        assert "a must be between [13.0, 14.0)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, fail upper
            threshold_model.check_bounds("a", "[9, 11)")
        assert "a must be between [9.0, 11.0)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, inclusive upper, fail
            threshold_model.check_bounds("a", "[9, 12.5)")
        assert "a must be between [9.0, 12.5)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, inclusive upper, fail
            threshold_model.check_bounds("d", "[9, 13)")
        assert "All elements of d must be between [9.0, 13.0)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail lower
            threshold_model.check_bounds(["a", "b"], "[11, 13)")
        assert "b must be between [11.0, 13.0)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail upper
            threshold_model.check_bounds(["a", "b"], "[9, 11)")
        assert "a must be between [9.0, 11.0)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, inclusive upper, fail
            threshold_model.check_bounds(["a", "b"], "[9, 12.5)")
        assert "a must be between [9.0, 12.5)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, inclusive both, fail
            threshold_model.check_bounds(["a", "b"], "[10, 12.5)")
        assert "a must be between [10.0, 12.5)" == str(excinfo.value)

    def test_threshold_within_bounds_exclusive_both(self, threshold_model):
        with not_raises(ValueError):
            # single value
            threshold_model.check_bounds("a", "(9, 13)")
            # multiple value
            threshold_model.check_bounds(["a", "b"], "(9, 13)")
            # multiple value
            threshold_model.check_bounds(["a", "b", "d"], "(9, 13.5)")

        with pytest.raises(ValueError) as excinfo:
            # single value, fail lower
            threshold_model.check_bounds("a", "(13, 14)")
        assert "a must be between (13.0, 14.0)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, fail upper
            threshold_model.check_bounds("a", "(9, 11)")
        assert "a must be between (9.0, 11.0)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # single value, inclusive lower, fail
            threshold_model.check_bounds("a", "(12.5, 13)")
        assert "a must be between (12.5, 13.0)" == str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            # single value, inclusive upper, fail
            threshold_model.check_bounds("a", "(9, 12.5)")
        assert "a must be between (9.0, 12.5)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail lower
            threshold_model.check_bounds(["a", "b"], "(11, 13)")
        assert "b must be between (11.0, 13.0)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, fail upper
            threshold_model.check_bounds(["a", "b"], "(9, 11)")
        assert "a must be between (9.0, 11.0)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, inclusive lower, fail
            threshold_model.check_bounds(["a", "b"], "(10, 13)")
        assert "b must be between (10.0, 13.0)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, inclusive upper, fail
            threshold_model.check_bounds(["a", "b"], "(9, 12.5)")
        assert "a must be between (9.0, 12.5)" == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            # multiple value, inclusive both, fail
            threshold_model.check_bounds(["a", "b"], "(10, 12.5)")
        # Implementation quirk: checks all lower bounds first and then upper
        # bound instead of checking both bounds following the key order
        assert "b must be between (10.0, 12.5)" == str(excinfo.value)

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
