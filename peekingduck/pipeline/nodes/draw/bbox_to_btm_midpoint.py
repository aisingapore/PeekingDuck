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

import numpy as np
from typing import Dict, List, Tuple

from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    def __init__(self, config: Dict) -> None:
        super().__init__(config, name='heuristic.bbox_to_btm_midpoint')

    def run(self, inputs: Dict) -> List[Tuple[float]]:
        """Converts bounding boxes to  a single point of reference
        for use in zone analytics

        Args:
            bboxes (List[List[float]]): List of bboxes, each a list of 4 elements that
            defines the bounding box 

        Returns:
            bbox_pts (List[tuple[float]]): List of x, y coordinates of the mid lower pt
            of each bounding box

        """

        # get xy midpoint of each bbox (x1, y1, x2, y2)
        # This is calculated as x is (x1-x2)/2 and y is y2
        bboxes = inputs[self.inputs[0]]
        frame = inputs[self.inputs[1]]
        self.img_size = (frame.shape[1], frame.shape[0])
        return {'btm_midpoint': [self._xy_on_img(((bbox[0]+bbox[2])/2), bbox[3])
                for bbox in bboxes]}

    def _xy_on_img(self, x, y):
        return (int(x * self.img_size[0]), int(y * self.img_size[1]))