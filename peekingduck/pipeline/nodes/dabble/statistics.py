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
A flexible node created for calculating the average, minimum and maximum of a single target
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
    """A flexible node created for calculating the average, minimum and maximum of a single target
    variable of interest over time. Supports in-built PeekingDuck data types defined in
    :doc:`Glossary </glossary>` as well as custom data types produced by custom nodes.

    The 3 examples below illustrate different ways of using this node. The ``target`` and ``apply``
    configs are used to extract the target variable of interest, which is for the current frame.
    The average is then re-computed with the new current value. This current value is also compared
    with the previous maximum value, and if higher, the maximum value is updated. The same applies
    for the minimum value.

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
    | Resulting Current Value       | 8             | 2                   | 1                              |
    |                               |               | (len[4,5] is 2)     | (only 52 >= 30)                |
    +-------------------------------+---------------+---------------------+--------------------------------+
    | Configs                                                                                              |
    +--------+----------------------+---------------+---------------------+--------------------------------+
    | target | data_type            | :mod:`count`  | :mod:`large_groups` | :mod:`obj_attrs`               |
    +        +----------------------+---------------+---------------------+--------------------------------+
    |        | get                  | null          | null                | ["details", "age"]             |
    +--------+----------------------+---------------+---------------------+--------------------------------+
    | apply  | function             | identity      | len                 | conditional_count              |
    |        +-----------+----------+---------------+---------------------+--------------------------------+
    |        | condition | operator | null          | null                | ">="                           |
    |        |           +----------+---------------+---------------------+--------------------------------+
    |        |           | operand  | null          | null                | 30                             |
    +--------+-----------+----------+---------------+---------------------+--------------------------------+

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

            - data_type (:obj:`str`): **default=""** |br|
                The input data type from a prior node in the pipeline, such as ``obj_attrs``.
                Only node inputs that are of type ``int``, ``List`` or ``Dict`` are accepted.
                Aside from in-built PeekingDuck data types, it is also possible to use
                custom data types produced by custom nodes.

            - get (:obj:`Optional[List]`): **default=null** |br|
                If ``node_output`` is of type ``Dict``, the key of interest of the dictionary
                are to be included here. Using the "Dict example" above where the target of
                interest is "age". List the keys of the ``obj_attrs`` dictionary required to get the desired tag. For
                example 2, to draw the tag given by the attribute `"ids"`, the required key
                is `["ids"]`. To draw the tag given by the attribute `"age"`, as `"age"` is nested
                within `"details"`, the required keys are `["details", "age"]`.


        apply (:obj:`Dict`) |br|

            - function (:obj:`str`): **{"identity", "len", "max", "conditional_count"}, default="identity"** |br|
                The function used to obtain the target variable from the original data type input.

                +-------------------+------------------+---------------------------------------------+
                | function          | valid data types | action                                      |
                +-------------------+------------------+---------------------------------------------+
                | identity          | int, float       | returns the same                            |
                +-------------------+------------------+---------------------------------------------+
                | len               | list, dict       | length of list or dict                      |
                +-------------------+------------------+---------------------------------------------+
                | max               | list             | max value of a list                         |
                +-------------------+------------------+---------------------------------------------+
                | conditional_count | list             | counts the number of occurrences within the |
                |                   |                  |                                             |
                |                   |                  | list that satisfies conditions stated in    |
                |                   |                  |                                             |
                |                   |                  | condition config                            |
                +-------------------+------------------+---------------------------------------------+

            - condition (:obj:`Dict`): **default = {"operator": null, "operand": null}**. |br|
                Optional application of condition for comparing the attribute
                with the ``operand`` using ``operator``. To be used if conditional
                count. Operators are ``==``, ``>``, ``>=``, ``<``, ``<=``.


    TO DO:
    - Check that curr value is of type int or float.
    - Write the above in instructions.

    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.avg, self.min, self.max = 0, 0, 0
        self.num_iter = 0

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Counts ...
        etc
        """
        self.num_iter += 1
        keys = deque(self.target["get"])
        attr_value = _get_value(inputs[self.target["data_type"]], keys)
        _check_data_type(attr_value)

        self.now = self._apply(
            attr_value, self.apply["function"], self.apply["condition"]
        )

        if self.now < self.min:
            self.min = self.now
        if self.now > self.max:
            self.max = self.now
        # https://ubuntuincident.wordpress.com/2012/04/25/calculating-the-average-incrementally/
        self.avg = (self.avg * self.num_iter + self.now) / (self.num_iter + 1)

        # only for prototyping
        if not self.num_iter % 10:
            print(
                f"avg: {self.avg:.2f}, min: {self.min}, max: {self.max}, now: {self.now}"
            )

        return {"avg": self.avg, "min": self.min, "max": self.max}

    def _apply(self, attr_value, function, condition):
        if function == "identity":
            return attr_value
        elif function == "len":
            if type(attr_value) == list or type(attr_value) == dict:
                return len(attr_value)
            raise ValueError("Type has to ...")
        elif function == "max":
            if attr_value and (type(attr_value) == list or type(attr_value) == dict):
                return max(attr_value)
            else:
                return 0
        elif function == "conditional_count":
            return _conditional_count(attr_value, condition)
        # this method cannot return None in all scenarios
        # check input_name is in datapool + check it's of type int, list or dict


def _get_value(data: Dict[str, Any], keys: deque) -> List[Union[str, int]]:
    if not keys:
        return data
    key = keys.popleft()
    return _get_value(data[key], keys)


def _check_data_type(attr_value):
    if (
        type(attr_value) != int
        and type(attr_value) != list
        and type(attr_value) != dict
    ):
        raise ValueError("it has to be of type int, list, dict...")


def _conditional_count(attr_value, condition):
    operator, operand = condition["operator"], condition["operand"]
    count = 0
    op_func = OPS[operator]
    for item in attr_value:
        if op_func(item, operand):
            count += 1
    return count
