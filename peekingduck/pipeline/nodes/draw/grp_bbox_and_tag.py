"""
Copyright 2021 AI Singapore

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Any, Dict, List

import numpy as np
from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.draw.utils.drawfunctions import draw_bboxes, draw_tags


class Node(AbstractNode):
    """This node draws large bounding boxes over multiple object bounding boxes
    which have been identified as belonging to large groups."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)
        self.tag = config["tag"]
        self.bbox_color = tuple(config["bbox_color"])
        self.bbox_thickness = config["bbox_thickness"]

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """ Draws large bounding boxes over multiple object bounding boxes
        which have been identified as belonging to large groups

        Args:
            inputs (dict): Dict with keys "img", "bboxes", "obj_groups",
                "large_groups".

        Returns:
            outputs (dict): Dict with keys "img".
        """

        group_bboxes = self._get_group_bbox_coords(
            inputs["large_groups"], inputs["bboxes"], inputs["obj_groups"])
        group_tags = self._get_group_tags(
            inputs["large_groups"], self.tag)

        draw_bboxes(inputs["img"], group_bboxes,
                    self.bbox_color, self.bbox_thickness)
        draw_tags(inputs["img"], group_bboxes,
                  group_tags, self.bbox_color)

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
        for group in large_groups:
            group_bbox = np.array([1., 1., 0., 0.])
            for idx, bbox in enumerate(bboxes):
                if obj_groups[idx] == group:
                    # outermost coord is < top left coord
                    if bbox[0] < group_bbox[0]:
                        group_bbox[0] = bbox[0]
                    if bbox[1] < group_bbox[1]:
                        group_bbox[1] = bbox[1]
                    # outermost coord is > bottom right coord
                    if bbox[2] > group_bbox[2]:
                        group_bbox[2] = bbox[2]
                    if bbox[3] > group_bbox[3]:
                        group_bbox[3] = bbox[3]
            group_bboxes.append(group_bbox)

        return group_bboxes

    @staticmethod
    def _get_group_tags(large_groups: List[int], tag: str) -> List[str]:
        """ Creates a list of tags to be used for the draw_tags function.
        """

        group_tags = []
        for _ in range(len(large_groups)):
            group_tags.append(tag)

        return group_tags
