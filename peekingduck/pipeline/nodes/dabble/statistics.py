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
A flexible node created for calculating the cumulative average, minimum and maximum of a single
variable of interest over time.
"""

import operator
from typing import Any, Dict, Union

from peekingduck.pipeline.nodes.dabble.statisticsv1 import utils
from peekingduck.pipeline.nodes.node import AbstractNode

# Order matters so that regex doesn't read ">=" as ">" or "<=" as "<"
# Dictionaries are insertion ordered from Python 3.6 onwards
OPS = {
    ">=": operator.ge,
    ">": operator.gt,
    "==": operator.eq,
    "<=": operator.le,
    "<": operator.lt,
}


class Node(AbstractNode):  # pylint: disable=too-many-instance-attributes
    """A flexible node created for calculating the cumulative average, minimum and maximum of a
    single variable of interest (defined as ``current result`` here) over time. The configurations
    for this node offer several methods to reduce the incoming data type into a single
    ``current result`` of type :obj:`int` or :obj:`float`, which is valid for the current
    video frame. ``current result`` is then used to recalculate the values of cumulative average,
    minimum, and maximum for PeekingDuck's running duration thus far.

    The configurations for this node are in the form ``<method>``: ``<expression>``. By default,
    all available methods have the ``null`` value for their ``<expression>`` values. To start off,
    **only one** ``<method>`` should be used, and you would choose it by replacing ``null`` with a
    valid ``<expression>`` string. More information about each ``<method>`` is provided in the
    **Configs** section below.

    As for ``<expression>`` string, it should be provided in the order ``<target attribute>
    <operator> <operand>``, where:

    * ``<target attribute>``: The input data type of interest, and dictionary keys (if required) \
    within square brackets (``[]``). Both in-built PeekingDuck data types defined in \
    :doc:`Glossary </glossary>` as well as custom data types produced by custom nodes are \
    supported.

    * ``<operator>`` (optional, only used for ``cond_count`` method): The operator used \
    for comparison, out of these 5 options: ``==``, ``>``, ``>=``, ``<``, ``<=``.

    * ``<operand>`` (optional, only used for ``cond_count`` method): The operand used for \
    comparison. If it is intended to be of :obj:`str` type, it should be enclosed by single or \
    double quotes. Else, it will be assumed to be of types :obj:`int` or :obj:`float`, and \
    converted into a :obj:`float` type for calculations.

    The examples in the table below illustrate how ``<method>``: ``<expression>`` choices reduce
    the incoming data type into the ``current result``.

    +---------------------------------------+---------------------------------+-------------+
    | **<data type>: <value>**              | ``<method>``: ``<expression>``  | ``<current  |
    |                                       |                                 | result>``   |
    +---------------------------------------+---------------------------------+-------------+
    | count: 8                              | identity: count                 | 8           |
    +---------------------------------------+---------------------------------+-------------+
    | obj_attrs: {                          | length: obj_attrs["ids"]        | 3           |
    |                                       +---------------------------------+-------------+
    |                                       | maximum:                        | 52          |
    |   ids: [1,2,4],                       | obj_attrs["details"]["age"]     |             |
    |                                       +---------------------------------+-------------+
    |   details: {                          | cond_count:                     | 2           |
    |                                       | obj_attrs["details"]["gender"]  |             |
    |     gender: ["male","male","female"], | == "male"                       |             |
    |                                       +---------------------------------+-------------+
    |     age: [52,17,48] }}                | cond_count:                     | 3           |
    |                                       | obj_attrs["details"]["age"]     |             |
    |                                       | < 60                            |             |
    +---------------------------------------+---------------------------------+-------------+

    Inputs:
        |all_input|

    Outputs:
        |cum_avg|

        |min|

        |max|

    Configs:
        identity (:obj:`str`): **default=null** |br|
            Accepts ``<target attribute>`` of types :obj:`int` or :obj:`float`, and returns the
            same value.
        length (:obj:`str`): **default=null** |br|
            Accepts ``<target attribute>`` of types :obj:`List[Any]` or :obj:`Dict[str, Any]`,
            and returns its length.
        minimum (:obj:`str`): **default=null** |br|
            Accepts ``<target attribute>`` of types :obj:`List[float | int]` or
            :obj:`Dict[str, float | int]`, and returns the minimum element within.
        maximum (:obj:`str`): **default=null** |br|
            Accepts ``<target attribute>`` of types :obj:`List[float | int]` or
            :obj:`Dict[str, float | int]`, and returns the maximum element within.
        cond_count (:obj:`str`): **default=null** |br|
            Accepts ``<target attribute>`` of types :obj:`List[float | int | str]`, and compares
            each element with ``<operand>`` using the ``<operator>``. The number of elements that
            fulfil the condition are counted towards ``<current result>``.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.cum_avg, self.min, self.max = 0.0, float("inf"), 0
        self.num_iter = 0
        all_methods = {
            "cond_count": self.cond_count,
            "identity": self.identity,
            "length": self.length,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }
        self.stats = utils.Stats(OPS)
        self.data_type, self.keys = self.stats.prepare_data(all_methods)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates the average, minimum and maximum of a single variable of interest over time.

        Args:
            inputs (dict): Dictionary with all available keys.

        Returns:
            outputs (dict): Dictionary with keys "cum_avg", "min" and "max".
        """

        self.curr = self.stats.get_curr_result(inputs[self.data_type], self.keys.copy())

        # if no detections in this frame, return stats from previous detections
        if not self.curr:
            return {"cum_avg": self.cum_avg, "min": self.min, "max": self.max}

        self.num_iter += 1
        self._update_stats(self.curr)

        return {"cum_avg": self.cum_avg, "min": self.min, "max": self.max}

    def _update_stats(self, curr: Union[float, int]) -> None:
        """Updates the cum_avg, min and max values with the current value."""
        if not isinstance(curr, int) and not isinstance(curr, int):
            raise ValueError(
                f"The current value has to be of type 'int' or 'float' to calculate statistics."
                f"However, the current value here is: '{curr}' which is of type: {type(curr)}."
            )
        if curr < self.min:
            self.min = curr
        if curr > self.max:
            self.max = curr
        self.cum_avg = (self.cum_avg * self.num_iter + curr) / (self.num_iter + 1)
