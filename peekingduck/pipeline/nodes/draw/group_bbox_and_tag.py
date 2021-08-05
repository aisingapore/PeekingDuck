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
Draws detected groups and their tags
"""

from typing import Any, Dict, List

import numpy as np
from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.draw.utils.bbox import draw_bboxes, draw_tags
from peekingduck.pipeline.nodes.draw.utils.constants import TOMATO


class Node(AbstractNode):
    """This node draws large bounding boxes over multiple object bounding boxes
    which have been identified as belonging to large groups.

    The draw group_bbox_and_tage node uses the obj_groups and large_groups from the
    dabble nodes to draw group bboxes and the large group message tag onto the image.
    For better understanding, refer to
    `group size checking usecase <use_cases/group_size_checking.html>`_.

    Inputs:

        |img|

        |bboxes|

        |obj_groups|

        |large_groups|

    Outputs:
        |none|

    Configs:
        tag (:obj:`str`): **default = "LARGE GROUP!"**
            string message printed in the case of a large group detected.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """ Draws large bounding boxes over multiple object bounding boxes
        which have been identified as belonging to large groups

        Args:
            inputs (dict): Dict with keys "img", "bboxes", "obj_groups",
                "large_groups".

        Returns:
            outputs (dict): Dict with keys "none".
        """

        group_bboxes = self._get_group_bbox_coords(
            inputs["large_groups"], inputs["bboxes"], inputs["obj_groups"])
        group_tags = self._get_group_tags(
            inputs["large_groups"], self.tag)

        # show labels set to False to reduce clutter on display
        draw_bboxes(inputs["img"], group_bboxes, TOMATO, False)   # type: ignore
        draw_tags(inputs["img"], group_bboxes, group_tags, TOMATO)  # type: ignore

        return {}

    @staticmethod
    def _get_group_bbox_coords(large_groups: List[int],
                               bboxes: List[np.array],
                               obj_groups: List[int]) -> List[np.array]:
        """ For bboxes that belong to the same large group, get the coordinates of
        a large bbox that combines all these individual bboxes. Repeat for all large
        groups.
        """
        group_bboxes = []
        bboxes = np.array(bboxes)
        obj_groups = np.array(obj_groups)
        for group in large_groups:
            # filter relevant bboxes, select top-left and bot-right corners
            group_bbox = np.array([1., 1., 0., 0.])
            selected_bboxes = bboxes[obj_groups == group]
            group_bbox[:2] = np.amin(selected_bboxes, axis=0)[:2]
            group_bbox[2:] = np.amax(selected_bboxes, axis=0)[2:]
            group_bboxes.append(group_bbox)

        return group_bboxes

    @staticmethod
    def _get_group_tags(large_groups: List[int], tag: str) -> List[str]:
        """ Creates a list of tags to be used for the draw_tags function.
        """

        group_tags = [tag for _ in range(len(large_groups))]

        return group_tags
