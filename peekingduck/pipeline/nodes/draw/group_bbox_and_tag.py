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
Draws large bounding boxes with tags, over identified groups of bounding boxes.
"""

from typing import Any, Dict, List

import numpy as np

from peekingduck.pipeline.nodes.draw.utils.bbox import draw_bboxes, draw_tags
from peekingduck.pipeline.nodes.draw.utils.constants import TOMATO
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """Draws large bounding boxes with tags over multiple object bounding boxes
    which have been identified as belonging to the same group.

    The :term:`large_groups` data type from :mod:`dabble.check_large_groups`,
    and the ``groups`` key of the :term:`obj_attrs` data type from
    :mod:`dabble.group_nearby_objs`, are inputs to this node which identify the
    different groups, and the group associated with each bounding box.

    For better understanding, refer to
    the :doc:`Group Size Checking use case </use_cases/group_size_checking>`.

    Inputs:
        |img_data|

        |bboxes_data|

        |obj_attrs_data|

        |large_groups_data|

    Outputs:
        |none_output_data|

    Configs:
        tag (:obj:`str`): **default = "LARGE GROUP!"**. |br|
            The string message printed when a large group is detected.

    .. versionchanged:: 1.2.0
        :mod:`draw.group_bbox_and_tag` used to take in ``obj_tags``
        (:obj:`List[str]`) as an input data type, which has been deprecated and
        now subsumed under :term:`obj_attrs`. The same attribute is accessed by
        the ``groups`` key of :term:`obj_attrs`.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Draws large bounding boxes over multiple object bounding boxes
        which have been identified as belonging to large groups.

        Args:
            inputs (dict): Dictionary with keys "img", "bboxes", "obj_attrs",
                "large_groups".

        Returns:
            outputs (dict): Dictionary with keys "none".
        """
        group_bboxes = self._get_group_bbox_coords(
            inputs["large_groups"], inputs["bboxes"], inputs["obj_attrs"]["groups"]
        )
        group_tags = self._get_group_tags(inputs["large_groups"], self.tag)

        # show labels set to False to reduce clutter on display
        draw_bboxes(inputs["img"], group_bboxes, [], False, TOMATO)
        draw_tags(inputs["img"], group_bboxes, group_tags, TOMATO)

        return {}

    @staticmethod
    def _get_group_bbox_coords(
        large_groups: List[int], bboxes: List[np.ndarray], obj_groups: List[int]
    ) -> List[np.ndarray]:
        """For bboxes that belong to the same large group, get the coordinates of
        a large bbox that combines all these individual bboxes. Repeat for all large
        groups.
        """
        group_bboxes = []
        bboxes = np.array(bboxes)
        obj_groups = np.array(obj_groups)
        for group in large_groups:
            # filter relevant bboxes, select top-left and bot-right corners
            group_bbox = np.array([1.0, 1.0, 0.0, 0.0])
            selected_bboxes = bboxes[obj_groups == group]
            group_bbox[:2] = np.amin(selected_bboxes, axis=0)[:2]
            group_bbox[2:] = np.amax(selected_bboxes, axis=0)[2:]
            group_bboxes.append(group_bbox)

        return group_bboxes

    @staticmethod
    def _get_group_tags(large_groups: List[int], tag: str) -> List[str]:
        """Creates a list of tags to be used for the draw_tags function."""
        group_tags = [tag for _ in range(len(large_groups))]

        return group_tags
