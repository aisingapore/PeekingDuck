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
Draws bounding boxes over detected objects.
"""

from typing import Any, Dict

from peekingduck.nodes.abstract_node import AbstractNode
from peekingduck.nodes.draw.utils.bbox import draw_bboxes


class Node(AbstractNode):
    """Draws bounding boxes on image.

    The :mod:`draw.bbox` node uses :term:`bboxes` and, optionally,
    :term:`bbox_labels` from the model predictions to draw the bbox predictions
    onto the image.

    Inputs:
        |img_data|

        |bboxes_data|

        |bbox_labels_data|

        |bbox_scores_data|

    Outputs:
        |none_output_data|

    Configs:
        show_labels (:obj:`bool`): **default = False**. |br|
            If ``True``, shows class label, e.g., "person", above the bounding
            box.
        show_scores (:obj:`bool`): **default = False**. |br|
            If ``True``, shows prediction scores, e.g., "0.99", at the bottom
            left of the bounding box.
        show colors (:obj:`List[int]`): **default = []**. |br|
            Define the color of the bounding box, in BGR format.
            Defined values have to be integers, and :math:`0 \\leq value \\leq 255`.

    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:

        # Optional Parameter
        color_choice = self.color_choice if 'color_choice' in self.__dict__ else None

        draw_bboxes(
            inputs["img"],
            inputs["bboxes"],
            inputs["bbox_labels"],
            self.show_labels,
            inputs["bbox_scores"],
            self.show_scores,
            color_choice,
        )
        return {}

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {"show_labels": bool, "show_scores": bool}
