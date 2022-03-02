# Copyright 2022 AI Singapore

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import operator
import pytest

from peekingduck.pipeline.nodes.dabble.statistics import Node
from peekingduck.pipeline.nodes.dabble.statisticsv1 import utils


@pytest.fixture
def stats_class():
    OPS = {
        ">=": operator.ge,
        ">": operator.gt,
        "==": operator.eq,
        "<=": operator.le,
        "<": operator.lt,
    }
    return utils.Stats(OPS)


@pytest.fixture
def all_methods():
    return {
        "identity": None,
        "minimum": None,
        "maximum": None,
        "length": None,
        "cond_count": None,
    }


@pytest.fixture
def stats_config():
    return {
        "input": ["all"],
        "output": ["cum_avg", "min", "max"],
        "identity": None,
        "minimum": None,
        "maximum": None,
        "length": None,
        "cond_count": None,
    }


@pytest.fixture(params=["identity", "minimum", "maximum", "length"])
def non_cond_method(request):
    yield request.param


@pytest.fixture(
    params=["count", " count ", "'count'", " 'count' ", "' count '", " ' count ' "]
)
def non_cond_without_key_expr(request):
    yield request.param


@pytest.fixture(
    params=[
        "obj_attrs['details']['age']",
        'obj_attrs["details"]["age"]',
        "obj_attrs[details][age]",
        " obj_attrs [ ' details ' ][ ' age ' ] ",
        ' obj_attrs [ " details " ][ " age " ] ',
        " obj_attrs [ details ] [ age ] ",
    ]
)
def non_cond_with_key_expr(request):
    yield request.param


@pytest.fixture(params=["obj_groups >= 35", "obj_groups>=35", "obj_groups>= 35 "])
def cond_without_key_expr(request):
    yield request.param


@pytest.fixture(
    params=[
        "obj_attrs['details']['age'] >= 35",
        "obj_attrs['details']['age']>=35",
        "obj_attrs['details']['age']>= 35 ",
    ]
)
def cond_with_key_expr(request):
    yield request.param


class TestDataTypeKeysCond:
    def test_non_cond_without_keys(
        self, stats_class, all_methods, non_cond_method, non_cond_without_key_expr
    ):
        all_methods[non_cond_method] = non_cond_without_key_expr
        data_type, keys = stats_class.prepare_data(all_methods)
        assert data_type == "count"
        assert keys == []

    def test_non_cond_with_keys(
        self, stats_class, all_methods, non_cond_method, non_cond_with_key_expr
    ):
        all_methods[non_cond_method] = non_cond_with_key_expr
        data_type, keys = stats_class.prepare_data(all_methods)
        assert data_type == "obj_attrs"
        assert keys == ["details", "age"]

    def test_cond_without_keys(self, stats_class, all_methods, cond_without_key_expr):
        all_methods["cond_count"] = cond_without_key_expr
        data_type, keys = stats_class.prepare_data(all_methods)
        assert data_type == "obj_groups"
        assert stats_class.condition["operand"] == 35.0
        assert stats_class.condition["op_func"] == operator.ge
        assert keys == []

    def test_cond_with_keys(self, stats_class, all_methods, cond_with_key_expr):
        all_methods["cond_count"] = cond_with_key_expr
        data_type, keys = stats_class.prepare_data(all_methods)
        assert data_type == "obj_attrs"
        assert stats_class.condition["operand"] == 35.0
        assert stats_class.condition["op_func"] == operator.ge
        assert keys == ["details", "age"]

    def test_cond_operators(
        self, stats_class, stats_config, all_methods, non_cond_method
    ):
        stats_config["cond_count"] = "obj_groups"
        with pytest.raises(ValueError) as excinfo:
            Node(stats_config)
        assert "should have an operator for comparison" in str(excinfo.value)
        stats_config["cond_count"] = None

        stats_config[non_cond_method] = "obj_groups >= 35"
        with pytest.raises(ValueError) as excinfo:
            Node(stats_config)
        assert "should not have" in str(excinfo.value)

        # Especially for checking that regex doesn't read ">=" as ">" or "<=" as "<"
        test_cases = {
            "obj_groups >= 35": operator.ge,
            "obj_groups > 35": operator.gt,
            "obj_groups == 35": operator.eq,
            "obj_groups <= 35": operator.le,
            "obj_groups < 35": operator.lt,
        }
        for expr, ans in test_cases.items():
            all_methods["cond_count"] = expr
            stats_class.prepare_data(all_methods)
            assert stats_class.condition["op_func"] == ans

    def test_cond_operands(self, stats_class, all_methods):
        test_cases = {
            "obj_groups >= 35": 35.0,
            "obj_groups >= '35'": "35",
            'obj_groups >= "35"': "35",
            "flags == 'TOO CLOSE!'": "TOO CLOSE!",
            "flags == ' TOO CLOSE! '": " TOO CLOSE! ",
        }
        for expr, ans in test_cases.items():
            all_methods["cond_count"] = expr
            stats_class.prepare_data(all_methods)
            assert stats_class.condition["operand"] == ans


class TestStatistics:
    def test_no_methods_chosen(self, stats_config):
        with pytest.raises(ValueError) as excinfo:
            Node(stats_config)
        assert "one method needs to be selected" in str(excinfo.value)

    def test_multiple_methods_chosen(self, stats_config):
        stats_config["identity"] = "count"
        stats_config["maximum"] = "ids"
        with pytest.raises(ValueError) as excinfo:
            Node(stats_config)
        assert "only one method should be selected" in str(excinfo.value)
