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
Utility functions for statistics
"""
import re
from typing import Any, Callable, Dict, List, Tuple, Union


class Stats:
    """
    Extract the chosen <method>: <expression> config and apply to incoming data.

    Args:
        ops (Dict[str, Callable]): A dictionary of operator symbols (keys) pointing to
        callable operator modules (values).
    """

    def __init__(self, ops: Dict[str, Callable]) -> None:
        self.ops = ops
        self.method: str
        self.expr: str
        self.condition = {"op_func": Callable, "operand": str}
        self.data_type: str
        self.keys: List[str] = []

    def prepare_data(
        self, all_methods: Dict[str, Union[str, None]]
    ) -> Tuple[str, List[str]]:
        """
        Does the necessary data preparation during node initialisation.

        Args:
            - all_methods (Dict[str, Union[str, None]]): Dictionary of <method>: <expression>.

        Returns:
            - self.data_type (str): PeekingDuck in-built data type or custom data type from custom
            nodes.
            - self.keys (List[str]): List of selected keys if self.data_type is a dictionary.
            If not, returns an empty list.
        """
        self.method, self.expr = _get_method_expr(all_methods)
        self._update_data_type_keys_condition()

        return self.data_type, self.keys

    def get_curr(self, inputs: Any, keys: List[str]) -> Union[float, int, None]:
        """
        Extracts the current resulting value from the incoming data.

        Args:
            - inputs (Any): Current value for the
            self.data_type key.
            - keys (List[str]): List of selected keys if self.data_type is a dictionary.

        Returns:
            - curr (Union[float, int, None]): Current resulting value.

        """
        target_attr = _get_value(inputs, keys)
        curr = _apply_method(target_attr, self.method, self.condition)

        return curr

    def _update_data_type_keys_condition(self) -> None:
        """Uses self.method, self.expr, and self.ops to update self.data_type, self.keys, and
        self.condition."""
        ops_expr = "(" + ")|(".join(self.ops.keys()) + ")"
        match = re.search(ops_expr, self.expr)
        if not match and self.method == "conditional_count":
            raise ValueError(
                f"The chosen method: {self.method} should have an operator for comparison."
            )
        if match and self.method != "conditional_count":
            raise ValueError(
                f"The chosen method: {self.method} should not have the {match.group()} "
                f"operator."
            )

        if match:
            self.condition["op_func"] = self.ops[match.group()]
            op_idx_start, op_idx_end = match.span()
            target_attr = self.expr[:op_idx_start]
            # check int/float/str for operand
            operand_raw = self.expr[op_idx_end + 1 :]
            self.condition["operand"] = _get_operand(operand_raw)  # type: ignore
        else:
            target_attr = self.expr

        self.data_type, self.keys = _get_data_type_and_keys(target_attr)


def _get_method_expr(all_methods: Dict[str, Union[str, None]]) -> Tuple[str, str]:
    """Gets the only non-null key value pair from a dictionary."""
    num_methods = 0
    for method, expr in all_methods.items():
        if expr:
            num_methods += 1
            selected_method, selected_expr = method, expr

    if num_methods < 1:
        raise ValueError(
            "No methods selected in config, but one method needs to be selected to proceed."
        )
    if num_methods > 1:
        raise ValueError(
            f"{num_methods} methods selected in config, but only one method should be selected "
            f"to proceed."
        )
    return selected_method, selected_expr


def _get_data_type_and_keys(target_attr: str) -> Tuple[str, List[str]]:
    """Extract the name of data_type and its associated keys in a list, if any."""
    target_attr = re.sub(r"'|\"", "", target_attr)
    target_attr = target_attr.strip()
    keys = re.findall(r"\[(.*?)\]", target_attr)
    if not keys:
        data_type = target_attr
        keys = []
    else:
        data_type = target_attr.split("[")[0]

    return data_type, keys


def _get_operand(operand_raw: str) -> Union[str, float]:
    """Removes leading and trailing spaces, and return as a string or float."""
    operand_raw = operand_raw.strip()
    is_string = re.findall("'|\"", operand_raw)
    if is_string:
        operand = re.sub("'|\"", "", operand_raw)
        return operand
    try:
        return float(operand_raw)
    except TypeError as error:
        raise TypeError(
            f"The detected operand here is: {operand_raw}."
            f"If the operand is intended to be a string, ensure that it is enclosed by single "
            f"or double quotes. Else, it is assumed to be of integer or float type, and will be "
            f"subsequently converted into a float."
        ) from error


def _get_value(data: Dict[str, Any], keys: List[str]) -> Union[int, float, list, dict]:
    """Recursively goes through the keys of a dictionary to obtain the final value."""
    if not keys:
        return data
    key = keys.pop(0)
    return _get_value(data[key], keys)


def _apply_method(
    target_attr: Any,
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


def _conditional_count(target_attr: Any, condition: Dict[str, Any]) -> int:
    """Counts the number of elements in a list given an operator and operand for comparison."""
    count = 0
    for item in target_attr:
        if condition["op_func"](item, condition["operand"]):
            count += 1
    return count
