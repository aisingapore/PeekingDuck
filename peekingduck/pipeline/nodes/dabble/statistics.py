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
                - node_output (:obj:`str`): **default="count"** |br|
                    The output of a prior node in the pipeline, such as ``obj_attrs``. Only node
                    outputs that are of type ``int``, ``List`` or ``Dict`` are accepted. Aside
                    from PeekingDuck's standard node outputs, it is also possible to use
                    customised node outputs from your own custom nodes.
                - dict_keys (:obj:`Optional[List]`): **default=null** |br|
                    If ``node_output`` is of type ``Dict``, the key of interest of the dictionary
                    are to be included here. Using the "Dict example" above where the target of
                    interest is "age":

                    >>> # 3. Dict example
                    >>> node_output: obj_attrs
                    >>> dict_keys: ["details", "age"]


        apply (:obj:`Dict`): |br|
            Attributes are the outputs of previous nodes, for now, this node supports
            the calculation of attributes that are of ``int`` or ``List`` types. The calc
            method chosen will create a count of the current desired value ``curr``, from
            which ``avg``, ``min`` and ``max`` can be calculated over time. Note that all methods
            are aimed to reduce the input to a single value, e.g. max()
            Create a table - ``int_count`` does... ``list_max`` applies max([..]) if nothing 0 |br|

                - target_type (:obj:`str`): **{"int", "list"}, default="list"** |br|
                    chap chap chap
                - function (:obj:`str`): **{"identity", "len", "min", "max"}, default="len"** |br|
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
        attr_value = _get_attr_info(self.attribute, 0, inputs)

        self.now = self._apply_calc_method(attr_value)

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

    def _apply_calc_method(self, attr_value):
        if self.calc_method == "int":
            return attr_value
        elif self.calc_method == "list_len":
            return len(attr_value)
        elif self.calc_method == "list_max":
            if attr_value:
                return max(attr_value)
            else:
                return 0
        elif self.calc_method == "list_conditional":
            return _conditional_count(attr_value, self.condition)
        # this method cannot return None in all scenarios
        # check input_name is in datapool + check it's of type int, list or dict


def _get_attr_info(attr, attr_idx, data_trunc):
    if attr_idx == len(attr) - 1:
        return data_trunc[attr[attr_idx]]
    else:
        return _get_attr_info(attr, attr_idx + 1, data_trunc[attr[attr_idx]])


def _conditional_count(attr_value, condition):
    operator, operand = condition["operator"], condition["operand"]
    count = 0
    op_func = OPS[operator]
    for item in attr_value:
        if op_func(item, operand):
            count += 1
    return count
