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
Draws the 2D boundaries of a zone
"""

from typing import Any, Dict
from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.draw.utils.zone import draw_zones


class Node(AbstractNode):
    """Draws the boundaries of each specified zone onto the image.

    The draw zones node uses the zones from the zone_count dabble node to
    draw a bounding box that represents the zone boundaries onto the image.

    Inputs:

        |img|

        |zones|

    Outputs:
        |none|

    Configs:
        None.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Draws the boundaries of each specified zone onto the image.

        Args:
            inputs (dict): Dict with keys "zones", "img".

        Returns:
            outputs (dict): Dict with keys "none".
        """

        draw_zones(inputs["img"], inputs["zones"])  # type: ignore
        return {}
