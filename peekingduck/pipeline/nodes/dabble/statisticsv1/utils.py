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
    Extract the chosen <unary_expr> or <cond_expr> config and apply to incoming data.

    Args:
        ops (Dict[str, Callable]): A dictionary of operator symbols (keys) pointing to
        callable operator modules (values).
    """

    def __init__(self, ops: Dict[str, Callable]) -> None:
        self.ops = ops
        self.func: str
        self.expr: str
        self.condition = {"op_func": Callable, "operand": str}
        self.data_type: str
        self.keys: List[str] = []
        self.func_map: Dict[str, Dict[str, Any]] = {
            "identity": {"types": (int, float), "func": _func_identity},
            "length": {"types": (dict, list), "func": _func_length},
            "minimum": {"types": (dict, list), "func": _func_minimum},
            "maximum": {"types": (dict, list), "func": _func_maximum},
            "cond_count": {"types": (list), "func": _func_cond_count},
        }

    def prepare_data(
        self, all_funcs: Dict[str, Union[str, None]]
    ) -> Tuple[str, List[str]]:
        """
        Does the necessary data preparation during node initialisation.

        Args:
            - all_funcs (Dict[str, Union[str, None]]): Dictionary of <func>: <user input>.

        Returns:
            - self.data_type (str): PeekingDuck in-built data type or custom data type from custom
            nodes.
            - self.keys (List[str]): List of selected keys if self.data_type is a dictionary.
            If not, returns an empty list.
        """
        self.func, self.expr = _get_func_expr(all_funcs)
        self._update_data_type_keys_condition()

        return self.data_type, self.keys

    def get_curr_result(self, inputs: Any, keys: List[str]) -> Union[float, int, None]:
        """
        Extracts the current resulting value from the incoming data.

        Args:
            - inputs (Any): Current value for the self.data_type key.
            - keys (List[str]): List of selected keys if self.data_type is a dictionary.

        Returns:
            - curr (Union[float, int, None]): Current resulting value.

        """
        target_attr = _deep_get_value(inputs, keys)
        curr = self._apply_func(target_attr)

        return curr

    def _update_data_type_keys_condition(self) -> None:
        """Uses self.func, self.expr, and self.ops to update self.data_type, self.keys, and
        self.condition."""
        ops_expr = "(" + ")|(".join(self.ops.keys()) + ")"
        match = re.search(ops_expr, self.expr)

        if not match and self.func == "cond_count":
            raise ValueError(
                f"The chosen function: {self.func} should have an operator for comparison."
            )
        if match and self.func != "cond_count":
            raise ValueError(
                f"The chosen function: {self.func} should not have the {match.group()} "
                f"operator."
            )

        if match:
            operator = match.group()
            self.condition["op_func"] = self.ops[operator]
            op_idx_start, op_idx_end = match.span()
            target_attr = self.expr[:op_idx_start]
            operand_raw = self.expr[op_idx_end:]
            self.condition["operand"] = _get_operand(operand_raw, operator)  # type: ignore
        else:
            target_attr = self.expr

        self.data_type, self.keys = _get_data_type_and_keys(target_attr)

    def _apply_func(self, target_attr: Any) -> Union[int, float, None]:
        """Applies a function and optional condition to a target attribute."""
        if not target_attr:
            return None

        _check_type(target_attr, self.func, self.func_map[self.func]["types"])
        args = (target_attr, self.condition)
        return self.func_map[self.func]["func"](*args)


def _get_func_expr(all_funcs: Dict[str, Union[str, None]]) -> Tuple[str, str]:
    """Gets the only non-null key value pair from a dictionary."""
    num_funcs = 0
    for func, expr in all_funcs.items():
        if expr:
            num_funcs += 1
            selected_func, selected_expr = func, expr

    if num_funcs < 1:
        raise ValueError(
            "No functions selected in config, but one function needs to be selected to proceed."
        )
    if num_funcs > 1:
        raise ValueError(
            f"{num_funcs} functions selected in config, but only one function should be selected "
            f"to proceed."
        )
    return selected_func, selected_expr


def _get_data_type_and_keys(target_attr: str) -> Tuple[str, List[str]]:
    """Extract the name of data_type and its associated keys in a list, if any."""
    target_attr = re.sub(r"'|\"| ", "", target_attr)
    keys = re.findall(r"\[(.*?)\]", target_attr)
    if not keys:
        data_type = target_attr
        keys = []
    else:
        data_type = target_attr.split("[")[0]

    return data_type, keys


def _get_operand(operand_raw: str, operator: str) -> Union[str, float]:
    """Removes leading and trailing spaces, and return as a string or float."""
    operand_raw = operand_raw.strip()
    is_string = re.findall("'|\"", operand_raw)
    if is_string:
        if operator != "==":
            raise ValueError(
                f"The detected operand is: {operand_raw} and detected operator is: {operator}. "
                f"The operand is assumed to be a string as it is enclosed by single or double "
                f"quotes, and for string operand, only the '==' operator should be used for "
                f"comparison."
            )
        operand = re.sub("'|\"", "", operand_raw)
        return operand
    try:
        return float(operand_raw)
    except ValueError as error:
        raise ValueError(
            f"The detected operand here is: {operand_raw}."
            f"If the operand is intended to be a string, ensure that it is enclosed by single "
            f"or double quotes. Else, it is assumed to be of integer or float type, and will be "
            f"subsequently converted into a float."
        ) from error


def _deep_get_value(
    data: Dict[str, Any], keys: List[str]
) -> Union[int, float, list, dict]:
    """Recursively goes through the keys of a dictionary to obtain the final value."""
    if not keys:
        return data
    key = keys.pop(0)
    return _deep_get_value(data[key], keys)


def _func_identity(target_attr: Any, *_: Any) -> Union[float, int]:
    """Function for returning the identity of an integer or float."""
    return target_attr


def _func_length(target_attr: Union[Dict[str, Any], List[Any]], *_: Any) -> int:
    """Function for returning the length of a dictionary or list."""
    return len(target_attr)


def _func_minimum(
    target_attr: Union[
        Dict[Union[float, int], Union[float, int]], List[Union[float, int]]
    ],
    *_: Any,
) -> Union[float, int]:
    """Function for returning the minimum element of a list or dictionary."""
    try:
        return min(target_attr)
    except ValueError as error:
        raise ValueError(
            "To use the 'minimum' function, all elements within target_attr has to be"
            " of type 'int' or 'float'."
        ) from error


def _func_maximum(
    target_attr: Union[
        Dict[Union[float, int], Union[float, int]], List[Union[float, int]]
    ],
    *_: Any,
) -> Union[float, int]:
    """Function for returning the maximum element of a list or dictionary."""
    try:
        return max(target_attr)
    except ValueError as error:
        raise ValueError(
            "To use the 'maximum' function, all elements within the target_attr has to be"
            " of type 'int' or 'float'."
        ) from error


def _func_cond_count(target_attr: Any, condition: Dict[str, Any]) -> int:
    """Function for counting the number of elements in a list, given an operator and operand
    for comparison."""
    count = 0
    for item in target_attr:
        if condition["op_func"](item, condition["operand"]):
            count += 1
    return count


def _check_type(
    target_attr: Union[int, float, list, dict], func: str, *types: type
) -> None:
    """Checks that a target attribute conforms to the defined type(s)."""
    correct = False
    for obj_type in types:
        if isinstance(target_attr, obj_type):
            correct = True
    if not correct:
        raise TypeError(
            f"For the chosen function: '{func}', valid target_attr types are: {types}. "
            f"However, this target_attr: {target_attr} is of type: {type(target_attr)}."
        )
