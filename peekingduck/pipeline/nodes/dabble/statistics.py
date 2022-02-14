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
import operator

from typing import Any, Dict

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
    variable of interest over time. As there are many ways to use this node, the 3 examples below
    will be used to aid subsequent explanations.

    >>> # 1. int example
    >>> {"count": 8}
    >>> # 2. List example
    >>> {"large_groups": [4, 5]}
    >>> # 3. Dict example
    >>> {"obj_attrs": {"ids":[1,2], "details": {"gender": ["female","male"], "age": ["52","17"]}}

    To note: |br|
    - This node focuses on calculating the statistics of target variables accumulated over time.
    To obtain instantaneous counts, it is more suitable to use the ``dabble.bbox_count`` node
    which produces the output ``count``, which in fact is used in the first example. |br|
    - While the PeekingDuck team will continue to expand the capabilities of this node, you may
    have a specific use case that is best tackled by adapting from this node to create your own
    custom node.

    Inputs:
        |all|

    Outputs:
        |avg|

        |min|

        |max|

    Configs:
        target (:obj:`Dict`): A dictionary containing information for the target variable
            of interest. |br|
                - data_type (:obj:`str`): **default="count"** |br|
                    The input data type from a prior node in the pipeline, such as ``obj_attrs``.
                    Only node inputs that are of type ``int``, ``List`` or ``Dict`` are accepted.
                    Aside from PeekingDuck's standard node outputs, it is also possible to use
                    customised node outputs from your own custom nodes.
                - dict_keys (:obj:`Optional[List]`): **default=null** |br|
                    If ``node_output`` is of type ``Dict``, the key of interest of the dictionary
                    are to be included here. Using the "Dict example" above where the target of
                    interest is "age":

                    >>> # 3. Dict example
                    >>> node_output: obj_attrs
                    >>> dict_keys: ["details", "age"]


        apply (:obj:`Dict`): |br|
            Applies functions to the target value. The calc
            method chosen will create a count of the current desired value, from
            which ``avg``, ``min`` and ``max`` can be calculated over time. Note that all methods
            are aimed to reduce the input to a single value, e.g. max()
            Create a table - ``int_count`` does... ``list_max`` applies max([..]) if nothing 0 |br|

                - function (:obj:`str`): **{"identity", "len", "max", "conditional_count"},
                    default="identity"** |br|
                - condition (:obj:`Dict`): **default = {"operator": null, "operand": null}**. |br|
                    Optional application of condition for comparing the attribute
                    with the ``operand`` using ``operator``. To be used if conditional
                    count.
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
        attr_value = _get_attr_info(
            inputs, self.target["data_type"], self.target["dict_keys"]
        )
        print("attr_value", attr_value)
        _check_data_type(attr_value)

        self.now = self._apply(
            attr_value, self.apply["function"], self.apply["condition"]
        )
        print("self.now", self.now)

        if self.now < self.min:
            self.min = self.now
        if self.now > self.max:
            self.max = self.now
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


def _get_attr_info(inputs, data_type, dict_keys):
    if not dict_keys:
        return inputs[data_type]
    return _seek_value(inputs[data_type], dict_keys.copy())


def _seek_value(data, dict_keys):
    if not dict_keys:
        return data
    else:
        key = dict_keys.pop(0)
        return _seek_value(data[key], dict_keys)


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
