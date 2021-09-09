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

from typing import Any, Dict, List
import cv2
import numpy as np
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):  # pylint: disable=too-few-public-methods


    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        mosaic_img = self.mosaic_bbox(inputs["bboxes"], inputs["img"])
        outputs = {"img": mosaic_img}
        return outputs

    def mosaic_bbox(self, bboxes: List[np.ndarray], image: np.ndarray) -> np.ndarray:
        height = image.shape[0]
        width = image.shape[1]

        for bbox in bboxes:
            x_1, y_1, x_2, y_2 = bbox
            x_1, x_2 = int(x_1 * width), int(x_2 * width)
            y_1, y_2 = int(y_1 * height), int(y_2 * height)

            # slice the image to get the area bounded by bbox
            bbox_image = image[y_1:y_2, x_1:x_2, :].copy()
            mosaic_bbox_image = self.mosaic(bbox_image)
            image[y_1:y_2, x_1:x_2, :] = mosaic_bbox_image

        return image

    @staticmethod
    def mosaic(image: np.ndarray, blocks=7) -> np.ndarray:
        (h, w) = image.shape[:2]
        xSteps = np.linspace(0, w, blocks + 1, dtype="int")
        ySteps = np.linspace(0, h, blocks + 1, dtype="int")
    
        for i in range(1, len(ySteps)):
            for j in range(1, len(xSteps)):
                startX = xSteps[j - 1]
                startY = ySteps[i - 1]
                endX = xSteps[j]
                endY = ySteps[i]
            
                roi = image[startY:endY, startX:endX]
                (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
                cv2.rectangle(image, (startX, startY), (endX, endY), (B, G, R), -1)
            
        return image        