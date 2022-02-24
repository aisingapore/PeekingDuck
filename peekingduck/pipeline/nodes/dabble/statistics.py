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
A flexible node created for calculating the average, minimum and maximum of a single variable of
interest over time.
"""

from collections import deque, OrderedDict
import operator
import re
from typing import Any, Dict, List, Tuple, Type, Union

from peekingduck.pipeline.nodes.node import AbstractNode

# Order matters so that regex doesn't read ">=" as ">" or "<=" as "<"
OPS = OrderedDict(
    {
        ">=": operator.ge,
        ">": operator.gt,
        "==": operator.eq,
        "<=": operator.le,
        "<": operator.lt,
    }
)


class Node(AbstractNode):
    """A flexible node created for calculating the average, minimum and maximum of a single
    variable of interest over time.

    The data received by this node is processed in 2 steps. Firstly, as this node receives the
    :obj:`all` input, the ``target`` config is used to narrow down the specific data type of
    interest and dictionary keys (if required). Both in-built PeekingDuck data types defined in
    :doc:`Glossary </glossary>` as well as custom data types produced by custom nodes are
    supported. We refer to the result of step 1 as `target attribute`.

    Secondly, the ``apply`` config is used to apply a function and condition (if required) to
    `target attribute`, to obtain a final result which we refer to here as the `current value`.
    This `current value` is then used to recalculate the values of average, minimum and maximum.
    Note that in order to calculate these statistics, *current value* must be of type :obj:`int`
    or :obj:`float`.

    The 3 examples below illustrate different ways of using this node.

    +-------------------------------+---------------+---------------------+--------------------------+
    | Type                          | int           | list                | dict                     |
    +-------------------------------+---------------+---------------------+--------------------------+
    | Example                       | count: 8      | large_groups: [4,5] | obj_attrs: {             |
    |                               |               |                     |                          |
    |                               |               |                     |                          |
    |                               |               |                     |   ids: [1,2],            |
    |                               |               |                     |                          |
    |                               |               |                     |   details: {             |
    |                               |               |                     |                          |
    |                               |               |                     |     mask: [True, False], |
    |                               |               |                     |                          |
    |                               |               |                     |     age: [52,17] }}      |
    +-------------------------------+---------------+---------------------+--------------------------+
    | Objective                     | The given     | Number of items in  | Number of occurrences    |
    |                               |               |                     | where ``age`` >= 30      |
    |                               | number itself | ``large_groups``    |                          |
    +-------------------------------+---------------+---------------------+--------------------------+
    |                                            `Configs`                                           |
    +--------+----------------------+---------------+---------------------+--------------------------+
    | target | data_type            | :mod:`count`  | :mod:`large_groups` | :mod:`obj_attrs`         |
    |        +----------------------+---------------+---------------------+--------------------------+
    |        | get                  | null          | null                | ["details", "age"]       |
    +--------+----------------------+---------------+---------------------+--------------------------+
    | apply  | function             | identity      | len                 | conditional_count        |
    |        +-----------+----------+---------------+---------------------+--------------------------+
    |        | condition | operator | null          | null                | ">="                     |
    |        |           +----------+---------------+---------------------+--------------------------+
    |        |           | operand  | null          | null                | 30                       |
    +--------+-----------+----------+---------------+---------------------+--------------------------+
    |                                            `Results`                                           |
    +-------------------------------+---------------+---------------------+--------------------------+
    | Step 1: `Target attribute`    | 8             | [4,5]               | [52,17]                  |
    | after ``target`` config       |               |                     |                          |
    +-------------------------------+---------------+---------------------+--------------------------+
    | Step 2: `Current value`       | 8             | 2                   | 1                        |
    | after ``apply`` config        |               | (len[4,5] is 2)     | (only 52 >= 30)          |
    +-------------------------------+---------------+---------------------+--------------------------+


    Inputs:
        |all_input|

    Outputs:
        |avg|

        |min|

        |max|

    The 3 examples in the table above illustrate how these configs can be used for different
    scenarios.

    Configs:
        target (:obj:`Dict`) |br|

            - data_type (:obj:`str`): **default=null** |br|
                The input data type from a preceding node in the pipeline, such as ``obj_attrs``.
                Aside from in-built PeekingDuck data types, it is also possible to use custom
                data types produced by custom nodes.

            - get (:obj:`Optional[List]`): **default=null** |br|
                If ``data_type`` is of type ``dict``, list the keys of its dictionary required to
                get the desired value. In the *dict* example, to get the current value for
                `"age"`, as `"age"` is nested within `"details"`, the required keys are
                `["details", "age"]`.


        apply (:obj:`Dict`) |br|

            - function (:obj:`str`): **{"identity", "len", "max", "conditional_count"}, \
                default="identity"** |br|
                The function used to further reduce the `target attribute` into `current value`.
                `Target attribute` has to fulfil type requirements listed below.

                +-------------------+--------------+-----------------------------------+
                | function          | Valid types  | Action                            |
                |                   | for `target  |                                   |
                |                   | attribute`   |                                   |
                +-------------------+--------------+-----------------------------------+
                | identity          | int, float   | returns the same value            |
                +-------------------+--------------+-----------------------------------+
                | len               | list, dict   | length of list or dict            |
                +-------------------+--------------+-----------------------------------+
                | min               | list, dict   | finds the minimum                 |
                +-------------------+--------------+-----------------------------------+
                | max               | list, dict   | finds the maximum                 |
                +-------------------+--------------+-----------------------------------+
                | conditional_count | list         | counts the number of occurrences  |
                |                   |              |                                   |
                |                   |              | within a list that satisfies the  |
                |                   |              |                                   |
                |                   |              | condition stated in the           |
                |                   |              |                                   |
                |                   |              | ``condition`` config              |
                +-------------------+--------------+-----------------------------------+

            - condition (:obj:`Dict`): **default = {"operator": null, "operand": null}**. |br|
                Only used if ``conditional_count`` is selected for the ``function`` config. It
                counts the number of elements within the `target attribute` list that satisfy
                the comparison with the ``operand`` using ``operator``. Supported operators are
                ``==``, ``>``, ``>=``, ``<``, ``<=``.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.avg, self.min, self.max = 0.0, float("inf"), 0
        self.num_iter = 0
        all_methods = {
            "conditional_count": self.conditional_count,
            "identity": self.identity,
            "length": self.length,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }
        self.method, self.expr = _get_method_expr(all_methods)

        self.data_type, self.keys, self.condition = _get_run_data(
            self.method, self.expr
        )

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates the average, minimum and maximum of a single variable of interest over time.

        Args:
            inputs (dict): Dictionary with all available keys.

        Returns:
            outputs (dict): Dictionary with keys "avg", "min" and "max".
        """
        target_attr = _get_value(inputs[self.data_type], self.keys.copy())
        self.curr = _apply_func(target_attr, self.method, self.condition)

        # if no detections in this frame, return stats from previous detections
        if not self.curr:
            return {"avg": self.avg, "min": self.min, "max": self.max}

        self.num_iter += 1
        self._update_stats(self.curr)

        return {"avg": self.avg, "min": self.min, "max": self.max}

    def _update_stats(self, curr: Union[float, int]) -> None:
        """Updates the avg, min and max values with the current value."""
        if not isinstance(curr, int) and not isinstance(curr, int):
            raise ValueError(
                f"The current value has to be of type 'int' or 'float' to calculate statistics."
                f"However, the current value here is: '{curr}' which is of type: {type(curr)}."
            )
        if curr < self.min:
            self.min = curr
        if curr > self.max:
            self.max = curr
        self.avg = (self.avg * self.num_iter + curr) / (self.num_iter + 1)


def _get_method_expr(all_methods: Dict[str, Union[str, None]]) -> Tuple[str, str]:
    """Gets the only non-null key value pair from a dictionary."""
    num_methods = 0
    for method, expr in all_methods.items():
        if expr:
            num_methods += 1
            selected_method, selected_expr = method, expr

    if num_methods < 1:
        raise ValueError(
            f"No methods selected in config, but one method needs to be selected to proceed."
        )
    elif num_methods > 1:
        raise ValueError(
            f"{num_methods} methods selected in config, but only one method should be selected "
            f"to proceed."
        )
    return selected_method, selected_expr


def _get_run_data(method: str, expr: str) -> Tuple[str, List[str], Dict[str, Any]]:

    """
    "obj_attrs[ids]"
    "obj_attrs[flags] == TOO CLOSE!"
    "large_groups >= 2"

    target_attr __ operator __ operand

    """
    ops_expr = "(" + ")|(".join(OPS.keys()) + ")"
    match = re.search(ops_expr, expr)
    if not match and method == "conditional_count":
        raise ValueError(
            f"The chosen method: {method} should have an operator for comparison."
        )
    if match and method != "conditional_count":
        raise ValueError(
            f"The chosen method: {method} should not have the {match.group()} "
            f"operator."
        )

    condition = {"op_func": None, "operand": None}
    if match:
        condition["op_func"] = OPS[match.group()]
        op_idx_start, op_idx_end = match.span()
        target_attr = expr[:op_idx_start]
        # check int/float/str for operand
        operand_raw = expr[op_idx_end + 1 :]
        condition["operand"] = _get_operand(operand_raw)
    else:
        target_attr = expr

    data_type, keys = _get_data_type_and_keys(target_attr)

    return data_type, keys, condition


def _get_data_type_and_keys(target_attr: str) -> Tuple[str, List[Union[str, None]]]:
    target_attr = re.sub("'|\"", "", target_attr)
    target_attr = target_attr.strip()
    keys = re.findall("\[(.*?)\]", target_attr)
    if not keys:
        data_type = target_attr
        keys = []
    else:
        data_type = target_attr.split("[")[0]

    return data_type, keys


def _get_operand(operand_raw: str) -> Union[str, float]:
    operand_raw = operand_raw.strip()
    is_string = re.findall("'|\"", operand_raw)
    if is_string:
        operand = re.sub("'|\"", "", operand_raw)
        return operand
    try:
        return float(operand_raw)
    except:
        raise TypeError(
            f"The detected operand here is: {operand_raw}."
            f"If the operand is intended to be a string, ensure that it is enclosed by single "
            f"or double quotes. Else, it is assumed to be of integer or float type, and will be "
            f"subsequently converted into a float."
        )


def _get_value(data: Dict[str, Any], keys: deque) -> Union[int, float, list, dict]:
    """Recursively goes through the keys of a dictionary to obtain the final value."""
    if not keys:
        return data
    key = keys.pop(0)
    return _get_value(data[key], keys)


def _apply_func(
    target_attr: Union[int, float, list, dict],
    method: str,
    condition: Dict[str, Any],
) -> Union[int, float, None]:
    """Applies a method and optional condition to a target attribute."""
    if not target_attr:
        return None
    if method == "identity":
        _check_type(target_attr, method, int, float)
        return target_attr
    if method == "length":
        _check_type(target_attr, method, dict, list)
        return len(target_attr)
    if method == "minimum":
        _check_type(target_attr, method, dict, list)
        return min(target_attr)
    if method == "maximum":
        _check_type(target_attr, method, dict, list)
        return max(target_attr)
    # if method == "conditional_count"
    _check_type(target_attr, method, list)
    return _conditional_count(target_attr, condition)


def _check_type(
    target_attr: Union[int, float, list, dict], method: str, *types: type
) -> None:
    """Checks that a target attribute conforms to the defined type(s)."""
    correct = False
    for obj_type in types:
        if isinstance(target_attr, obj_type):
            correct = True
    if not correct:
        raise ValueError(
            f"For the chosen method: '{method}', valid target attribute types are: {types}. "
            f"However, this target attribute: {target_attr} is of type: {type(target_attr)}."
        )


def _conditional_count(
    target_attr: Union[int, float, List[Any], Dict[str, Any]], condition: Dict[str, Any]
) -> int:
    """Counts the number of elements in a list given an operator and operand for comparison."""
    count = 0
    for item in target_attr:
        if condition["op_func"](item, condition["operand"]):
            count += 1
    return count


"""
Rules:

a. String operands need to be wrapped in single/double quotes
b. Spaces in the expression will be ignored except for string operands

Say that don't put stuff in square brackets, we using regex
Unit tests

1. All operators work
2. Don't accept multiple operators
3. Spaces not included
4. 4 nulls and only 1 mode not null
5. "'asdasd'" and '"asfgasf"'

"""
