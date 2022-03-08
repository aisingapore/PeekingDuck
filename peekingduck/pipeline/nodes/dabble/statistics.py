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
Calculates the cumulative average, minimum and maximum of a single variable of interest over time.
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
    """Calculates the cumulative average, minimum and maximum of a single variable of interest
    (defined as ``current result`` here) over time. The configurations for this node offer several
    methods to reduce the incoming data type into a single ``current result`` of type :obj:`int`
    or :obj:`float`, which is valid for the current video frame. ``current result`` is then used to
    recalculate the values of cumulative average, minimum, and maximum for PeekingDuck's running
    duration thus far.

    The configuration for this node is described below using a combination of the `Extended Brackus
    -Naur Form (EBNF) <https://en.wikipedia.org/wiki/Extended_Backus–Naur_form>`_ and `Augmented
    Brackus-Naur Form (ABNF) <https://en.wikipedia.org/wiki/Augmented_Backus–Naur_form>`_
    metasyntax. Concrete examples are provided later to faciliate understanding. ::

        pkd_data_type   = ? PeekingDuck built-in data types ?
                        e.g. count, large_groups, obj_attrs
        user_data_type  = ? user data types produced by custom nodes ?
                        e.g. my_var, my_attrs
        keys            = ? Keys if pkd_data_type or user_data_type is a dictionary ?
                        e.g. ["ids"], ["details"]["age"]
        data_with_keys  = pkd_data_type keys | user_data_type keys
        data_wo_keys    = pkd_data_type | user_data_type
        target_attr     = data_with_keys | data_wo_keys

        unary_function  = "identity" | "length" | "maximum" | "minimum"
        unary_expr      = unary_function ":" target_attr

        operator        = "==" | ">=" | "<=" | ">" | "<"
        numbers         = ? Python integers or floats ?
        numeric_op      = operator numbers

        strings         = ? Python strings enclosed by single or double quotes ?
        string_op       = "==" strings

        cond_expr       = "cond_count" ":" target_attr ( numeric_op | string_op )

        configuration   = unary_expr | cond_expr

    It should be noted that square brackets (``[]``) should only be included in ``<keys>``. The
    table below illustrates how configuration choices reduce the incoming data type into the
    ``<current result>``.

    # pylint: disable=line-too-long
    +---------------------------------------+-------------------+--------------------------------+-------------+
    | ``<pkd_data_type>``: value            | ``<target_attr>`` | ``<unary_expr>``               | ``<current  |
    |                                       |                   |                                | result>``   |
    | or                                    |                   | or                             |             |
    |                                       |                   |                                |             |
    | ``<user_data_type>``: value           |                   | ``<cond_expr>``                |             |
    +---------------------------------------+-------------------+--------------------------------+-------------+
    | count: 8                              | count             | identity: count                | 8           |
    +---------------------------------------+-------------------+--------------------------------+-------------+
    | obj_attrs: {                          | obj_attrs["ids"]  | length: obj_attrs["ids"]       | 3           |
    |                                       +-------------------+--------------------------------+-------------+
    |                                       | obj_attrs         | maximum:                       | 52          |
    |   ids: [1,2,4],                       | ["details"]       | obj_attrs["details"]["age"]    |             |
    |                                       | ["age"]           |                                |             |
    |   details: {                          +-------------------+--------------------------------+-------------+
    |                                       | obj_attrs         | cond_count:                    | 2           |
    |     gender: ["male","male","female"], | ["details"]       | obj_attrs["details"]["gender"] |             |
    |                                       | ["gender"]        |                                |             |
    |     age: [52,17,48] }}                |                   | == "male"                      |             |
    |                                       +-------------------+--------------------------------+-------------+
    |                                       | obj_attrs         | cond_count:                    | 3           |
    |                                       | ["details"]       | obj_attrs["details"]["age"]    |             |
    |                                       | ["age"]           |                                |             |
    |                                       |                   | < 60                           |             |
    +---------------------------------------+-------------------+--------------------------------+-------------+

    Inputs:
        |all_input|

    Outputs:
        |cum_avg|

        |cum_max|

        |cum_min|

    Configs:
        identity (:obj:`str`): **default=null** |br|
            Accepts ``<target attribute>`` of types :obj:`int` or :obj:`float`, and returns the
            same value.
        length (:obj:`str`): **default=null** |br|
            Accepts ``<target attribute>`` of types :obj:`List[Any]` or :obj:`Dict[str, Any]`,
            and returns its length.
        minimum (:obj:`str`): **default=null** |br|
            Accepts ``<target attribute>`` of types :obj:`List[float | int]` or
            :obj:`Dict[str, float | int]`, and returns the minimum element within for the current
            frame. Not to be confused with the ``cum_min`` output data type, which represents the
            cumulative minimum over time.
        maximum (:obj:`str`): **default=null** |br|
            Accepts ``<target attribute>`` of types :obj:`List[float | int]` or
            :obj:`Dict[str, float | int]`, and returns the maximum element within for the current
            frame. Not to be confused with the ``cum_max`` output data type, which represents the
            cumulative maximum over time.
        cond_count (:obj:`str`): **default=null** |br|
            Accepts ``<target attribute>`` of types :obj:`List[float | int | str]`, and compares
            each element with ``<operand>`` using the ``<operator>``. The number of elements that
            fulfil the condition are counted towards ``<current result>``.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.cum_avg, self.cum_min, self.cum_max = 0.0, float("inf"), float("-inf")
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
            outputs (dict): Dictionary with keys "cum_avg", "cum_min" and "cum_max".
        """

        self.curr = self.stats.get_curr_result(inputs[self.data_type], self.keys.copy())

        # if no detections in this frame, return stats from previous detections
        if not self.curr:
            return {
                "cum_avg": self.cum_avg,
                "cum_min": self.cum_min,
                "cum_max": self.cum_max,
            }

        self._update_stats(self.curr)

        return {
            "cum_avg": self.cum_avg,
            "cum_min": self.cum_min,
            "cum_max": self.cum_max,
        }

    def _update_stats(self, curr: Union[float, int]) -> None:
        """Updates the cum_avg, cum_min and cum_max values with the current value."""
        if not isinstance(curr, float) and not isinstance(curr, int):
            raise TypeError(
                f"The current value has to be of type 'int' or 'float' to calculate statistics."
                f"However, the current value here is: '{curr}' which is of type: {type(curr)}."
            )

        if curr < self.cum_min:
            self.cum_min = curr
        if curr > self.cum_max:
            self.cum_max = curr
        if self.num_iter == 0:
            self.cum_avg = curr
        else:
            self.cum_avg = (self.cum_avg * self.num_iter + curr) / (self.num_iter + 1)
        self.num_iter += 1
