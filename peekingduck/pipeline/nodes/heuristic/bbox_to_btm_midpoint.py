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
Converts bounding boxes to a single point of reference
"""


from typing import Dict, Any, Tuple
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """This node converts bounding boxes to a singel pt which is
    the bottom midpoint of the bounding box"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)
        self.img_size = None

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Converts bounding boxes to a single point of reference
        for use in zone analytics

        Args:
            inputs (dict): Dict with keys "bboxes" and "img".
        Returns:
            outputs (dict): Dict with keys "btm_midpoint".

        """
        # get xy midpoint of each bbox (x1, y1, x2, y2)
        # This is calculated as x is (x1-x2)/2 and y is y2
        bboxes = inputs["bboxes"]
        frame = inputs["img"]
        self.img_size = (frame.shape[1], frame.shape[0])  # type:ignore
        return {'btm_midpoint': [self._xy_on_img(((bbox[0]+bbox[2])/2), bbox[3])
                                 for bbox in bboxes]}

    def _xy_on_img(self, pt_x: float, pt_y: float) -> Tuple[int, int]:
        """Return the int x y points of the midpoint on the original image"""
        assert self.img_size is not None
        return (int(pt_x * self.img_size[0]), int(pt_y * self.img_size[1]))
