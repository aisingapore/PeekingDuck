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
Counts the number of unique object tags accumulated over time.
"""

from typing import Any, Dict

from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """Counts the number of unique object tags accumulated over time from object tracking.
    It assumes that each item within ``obj_tags`` is of type ``str(int)`` and the id of the first
    ever tag is "0". For example, ["0", "1", "2"] would give a count of 3.

    Inputs:
        |obj_tags|

    Outputs:
        |count|

    Configs:
        None.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.num_tags_over_time = 0

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Counts the number of unique object tags accumulated over time.

        Args:
            inputs (Dict): Inputs dictionary with the key `obj_tags`.

        Returns:
            (Dict): Outputs dictionary with the key `count`.
        """
        if not inputs["obj_tags"]:
            pass
        else:
            max_tag_in_frame = max(
                list(map(_convert_int_increment, inputs["obj_tags"]))
            )
            if max_tag_in_frame > self.num_tags_over_time:
                self.num_tags_over_time = max_tag_in_frame

        return {"count": self.num_tags_over_time}


def _convert_int_increment(tag: str) -> int:
    """Changes type to int and increment by 1 as tag ID begins with 0."""
    return int(tag) + 1
