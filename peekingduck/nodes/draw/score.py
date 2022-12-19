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
Draws the prediction score at the corner of the bounding box over detected objects.
"""

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from peekingduck.nodes.abstract_node import AbstractNode
from peekingduck.nodes.draw.utils.constants import(
    YELLOW,
    NORMAL_FONTSCALE,
    THICK
    )


class Node(AbstractNode):
    """Draws the prediction score of the detection near the bounding box.

    The :mod:`draw.score` node uses :term:`bbox_scores` from the model predictions 
    to draw the prediction score onto the image.
    
    Inputs:
        |img_data|

        |bboxes_data|

        |bbox_scores_data|

    Outputs:
        |none_output_data|
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._draw_scores(
            inputs["img"], inputs["bboxes"], inputs["bbox_scores"]
            )
        return {}


    def _draw_scores(self, img: np.array, bboxes: np.array, scores: np.array) -> None:
        """Draw the prediction scores on every bounding boxes
        
         for each bounding box:
            - compute (x1, y1) top-left, (x2, y2) bottom-right coordinates
            - convert score into a two decimal place numeric string
            - draw score string onto image using opencv's putText()
              (see opencv's API docs for more info)
        
        """

        image_size = (img.shape[1], img.shape[0])  # width, height

        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = self._map_bbox_to_image_coords(bbox, image_size)
            score = scores[i]
            score_str = f"{score:0.2f}"
            cv2.putText(img=img,text=score_str,
                org=(x1, y2),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=NORMAL_FONTSCALE,
                color=YELLOW,
                thickness=THICK,
            )

        return {}


    @staticmethod
    def _map_bbox_to_image_coords(bbox: List[float], image_size: Tuple[int, int]):
        """Map bounding box coords (from 0 to 1) to absolute image coords.
        
        Args:
            bbox (List[float]): relative coords x1, y1, x2, y2
            image_size (Tuple[int, int]): Width, Height of image

        Returns:
            List[int]: absolute image coords x1, y1, x2, y2
        """
        width, height = image_size[0], image_size[1]
        x1, y1, x2, y2 = bbox
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height

        return int(x1), int(y1), int(x2), int(y2)
    