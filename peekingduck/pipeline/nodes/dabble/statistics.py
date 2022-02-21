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
A flexible node created for calculating the average, minimum and maximum of a single 
variable of interest over time.
"""
from collections import deque
import operator

from typing import Any, Dict, List, Union

from peekingduck.pipeline.nodes.node import AbstractNode

OPS = {
    "==": operator.eq,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


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

    +-------------------------------+---------------+---------------------+--------------------------------+
    | Type                          | int           | list                | dict                           |
    +-------------------------------+---------------+---------------------+--------------------------------+
    | Example                       | count: 8      | large_groups: [4,5] | obj_attrs: {                   |
    |                               |               |                     |                                |
    |                               |               |                     |                                |
    |                               |               |                     |   ids: [1,2],                  |
    |                               |               |                     |                                |
    |                               |               |                     |   details: {                   |
    |                               |               |                     |                                |
    |                               |               |                     |     gender: ["female","male"], |
    |                               |               |                     |                                |
    |                               |               |                     |     age: [52,17] }}            |
    +-------------------------------+---------------+---------------------+--------------------------------+
    | Objective                     | The given     | Number of items in  | Number of occurrences where    |
    |                               |               |                     | ``age`` >= 30                  |
    |                               | number itself | ``large_groups``    |                                |
    +-------------------------------+---------------+---------------------+--------------------------------+
    |                                               `Configs`                                              |
    +--------+----------------------+---------------+---------------------+--------------------------------+
    | target | data_type            | :mod:`count`  | :mod:`large_groups` | :mod:`obj_attrs`               |
    |        +----------------------+---------------+---------------------+--------------------------------+
    |        | get                  | null          | null                | ["details", "age"]             |
    +--------+----------------------+---------------+---------------------+--------------------------------+
    | apply  | function             | identity      | len                 | conditional_count              |
    |        +-----------+----------+---------------+---------------------+--------------------------------+
    |        | condition | operator | null          | null                | ">="                           |
    |        |           +----------+---------------+---------------------+--------------------------------+
    |        |           | operand  | null          | null                | 30                             |
    +--------+-----------+----------+---------------+---------------------+--------------------------------+
    |                                               `Results`                                              |
    +-------------------------------+---------------+---------------------+--------------------------------+
    | Step 1: `Target attribute`    | 8             | [4,5]               | [52,17]                        |
    | after ``target`` config       |               |                     |                                |
    +-------------------------------+---------------+---------------------+--------------------------------+
    | Step 2: `Current value`       | 8             | 2                   | 1                              |
    | after ``apply`` config        |               | (len[4,5] is 2)     | (only 52 >= 30)                |
    +-------------------------------+---------------+---------------------+--------------------------------+


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

                +-------------------+------------------------------------+--------------------------------------------+
                | function          | Valid types for `target attribute` | Action                                     |
                +-------------------+------------------------------------+--------------------------------------------+
                | identity          | int, float                         | returns the same value                     |
                +-------------------+------------------------------------+--------------------------------------------+
                | len               | list, dict                         | length of list or dict                     |
                +-------------------+------------------------------------+--------------------------------------------+
                | min               | list, dict                         | finds the minimum                          |
                +-------------------+------------------------------------+--------------------------------------------+
                | max               | list, dict                         | finds the maximum                          |
                +-------------------+------------------------------------+--------------------------------------------+
                | conditional_count | list                               | counts the number of occurrences within a  |
                |                   |                                    |                                            |
                |                   |                                    | list that satisfies the condition stated   |
                |                   |                                    |                                            |
                |                   |                                    | in the ``condition`` config                |
                +-------------------+------------------------------------+--------------------------------------------+

            - condition (:obj:`Dict`): **default = {"operator": null, "operand": null}**. |br|
                Only used if ``conditional_count`` is selected for the ``function`` config. It
                counts the number of elements within the `target attribute` list that satisfy 
                the comparison with the ``operand`` using ``operator``. Supported operators are 
                ``==``, ``>``, ``>=``, ``<``, ``<=``.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.avg, self.min, self.max = 0, float("inf"), 0
        self.num_iter = 0

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates the average, minimum and maximum of a single variable of interest over time.

        Args:
            inputs (dict): Dictionary with all available keys.

        Returns:
            outputs (dict): Dictionary with keys "avg", "min" and "max".
        """
        target_attr = self._get_target_attr(inputs)
        self.curr = _apply_func(target_attr, self.apply)

        # if no detections in this frame, return stats from previous
        # detections
        if not self.curr:
            return {"avg": self.avg, "min": self.min, "max": self.max}

        self.num_iter += 1
        self._update_stats(self.curr)

        return {"avg": self.avg, "min": self.min, "max": self.max}

    def _get_target_attr(self, inputs):
        if self.target["get"]:
            keys = deque(self.target["get"])
            return _get_value(inputs[self.target["data_type"]], keys)
        return inputs[self.target["data_type"]]

    def _update_stats(self, curr):
        if type(curr) != int and type(curr) != float:
            raise ValueError(
                f"The current value has to be of type 'int' or 'float' to calculate statistics."
                f"However, the current value here is: '{curr}' which is of type: {type(curr)}."
            )
        if curr < self.min:
            self.min = curr
        if curr > self.max:
            self.max = curr
        self.avg = (self.avg * self.num_iter + curr) / (self.num_iter + 1)


def _get_value(data: Dict[str, Any], keys: List[str]) -> List[Union[str, int]]:
    if not keys:
        return data
    key = keys.popleft()
    return _get_value(data[key], keys)


def _apply_func(target_attr, apply):
    function = apply["function"]
    condition = apply["condition"]
    if function == "identity":
        _check_type(target_attr, function, int, float)
        return target_attr
    elif function == "len":
        _check_type(target_attr, function, dict, list)
        return len(target_attr)
    elif function == "min":
        _check_type(target_attr, function, dict, list)
        if not target_attr:
            return None
        return min(target_attr)
    elif function == "max":
        _check_type(target_attr, function, dict, list)
        if not target_attr:
            return None
        return max(target_attr)
    elif function == "conditional_count":
        _check_type(target_attr, function, list)
        return _conditional_count(target_attr, condition)


def _check_type(target_attr, function, *types):
    correct = False
    for obj_type in types:
        if type(target_attr) == obj_type:
            correct = True
    if not correct:
        raise ValueError(
            f"For the chosen function: '{function}', valid target attribute types are: {types}. "
            f"However, this target attribute: {target_attr} is of type: {type(target_attr)}."
        )


def _conditional_count(target_attr, condition):
    operator, operand = condition["operator"], condition["operand"]
    count = 0
    op_func = OPS[operator]
    for item in target_attr:
        if op_func(item, operand):
            count += 1
    return count
