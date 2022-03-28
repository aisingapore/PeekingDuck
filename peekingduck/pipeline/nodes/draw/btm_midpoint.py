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


"""Draws the bottom middle point of a bounding box."""

from typing import Any, Dict

from peekingduck.pipeline.nodes.draw.utils.bbox import draw_pts
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """The :mod:`draw.btm_midpoint` node uses :term:`bboxes` from the model
    predictions to draw the bottom midpoint of each bbox as a dot onto the
    image. For better understanding of the use case, refer to the
    :doc:`Zone Counting use case </use_cases/zone_counting>`.

    Inputs:
        |img_data|

        |btm_midpoint_data|

    Outputs:
        |none_output_data|

    Configs:
        None.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Draws btm midpoint of bounding bboxes.

        Args:
            inputs (dict): Dictionary with keys "bboxes".

        Returns:
            outputs (dict): Empty dictionary.
        """
        draw_pts(inputs["img"], inputs["btm_midpoint"])

        return {}
