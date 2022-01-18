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
A flexible node created for calculating the average, minimum, maximum and 
current counts of a single attribute.
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
    """A flexible node created for calculating the average, minimum, maximum and
    current counts of a single attribute, that can deal with a variety of inputs types.
    We will be expanding the number of options over time, if you require certain features you
    can create a custom node adapted from this node that serves your purpose.

    Inputs:
        |all|

    Outputs:
        To add to class.rst
        |avg|

        |min|

        |max|

        |curr|

    Configs:
        attribute (:obj:`List`): **default = ["count"]**. |br|
            Attributes are the outputs of previous nodes. If single attribute name
            like ``count``, [``count``], but if the attribute is in the form of
            dict, [``obj_tags``, ``flags``]. Link to glossary of input/output types.
        calc_method (:obj:`str`): **{"int", "list_len", "list_max", "list_conditional"},
            default="int_count"**. |br|
            Attributes are the outputs of previous nodes, for now, this node supports
            the calculation of attributes that are of ``int`` or ``List`` types. The calc
            method chosen will create a count of the current desired value ``curr``, from
            which ``avg``, ``min`` and ``max`` can be calculated over time.
            Create a table - ``int_count`` does... ``list_max`` applies max([..]) if nothing 0
        condition (:obj:`Dict`): **default = {"operator": null, "operand": null}**. |br|
            Optional application of condition for comparing the attribute
            with the ``operand`` using ``operator``. To be used if conditional
            count.

    Examples:


    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.avg, self.min, self.max, self.curr = 0, 0, 0, 0
        self.num_iter = 0

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Counts ...
        etc
        """
        self.num_iter += 1
        attr_value = _get_attr_info(self.attribute, 0, inputs)

        self.curr = self._apply_calc_method(attr_value)

        if self.curr < self.min:
            self.min = self.curr
        if self.curr > self.max:
            self.max = self.curr
        self.avg = (self.avg * self.num_iter + self.curr) / (self.num_iter + 1)

        if not self.num_iter % 10:
            print(
                f"avg: {self.avg:.2f}, min: {self.min}, max: {self.max}, curr: {self.curr}"
            )

        return {"avg": self.avg, "min": self.min, "max": self.max, "curr": self.curr}

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
