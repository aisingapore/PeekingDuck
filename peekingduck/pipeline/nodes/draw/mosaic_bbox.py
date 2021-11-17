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
Mosaics area bounded by bounding boxes over detected object
"""

from typing import Any, Dict, List

import cv2
import numpy as np

from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):  # pylint: disable=too-few-public-methods
    """Mosaics areas bounded by bounding boxes on image.

    The ``draw.mosaic_bbox`` node helps to anonymize detected objects by
    pixelating the areas bounded by bounding boxes in an image.

    Inputs:
        |img|

        |bboxes|

    Outputs:
        |img|

    Configs:
        mosaic_level (:obj:`int`): **default = 7**. |br|
            Defines the resolution of a mosaic filter (width |times| height).
            The number corresponds to the number of rows and columns used to
            create a mosaic. For example, the default setting
            (``mosaic_level = 7``) creates a 7 |times| 7 mosaic filter.
            Increasing the number increases the intensity of pixelation over an
            area.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.mosaic_level = self.config["mosaic_level"]

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        mosaic_img = self.mosaic_bbox(inputs["img"], inputs["bboxes"])
        outputs = {"img": mosaic_img}
        return outputs

    def mosaic_bbox(self, image: np.ndarray, bboxes: List[np.ndarray]) -> np.ndarray:
        """Mosaics areas bounded by bounding boxes on image.

        Args:
            image (:obj:`np.ndarray`): Input image.
            bboxes (:obj:`List[np.ndarray]`): A list of detected bboxes.

        Returns:
            image (:obj:`np.ndarray`): Image with mosaic applied over areas
            bounded by bounding boxes.
        """
        height = image.shape[0]
        width = image.shape[1]

        for bbox in bboxes:
            x_1, y_1, x_2, y_2 = bbox
            x_1, x_2 = int(x_1 * width), int(x_2 * width)
            y_1, y_2 = int(y_1 * height), int(y_2 * height)

            # slice the image to get the area bounded by bbox
            bbox_image = image[y_1:y_2, x_1:x_2, :].copy()
            mosaic_bbox_image = self.mosaic(bbox_image, self.mosaic_level)
            image[y_1:y_2, x_1:x_2, :] = mosaic_bbox_image

        return image

    @staticmethod
    def mosaic(  # pylint: disable-msg=too-many-locals
        image: np.ndarray, mosaic_level: int
    ) -> np.ndarray:
        """Mosaics a given input image.

        Args:
            image (:obj:`np.ndarray`): Input image.
            mosaic_level (:obj:`int`): Mosaic intensity.

        Returns:
            image (:obj:`np.ndarray`): Mosaiced image.
        """
        (height, width) = image.shape[:2]
        x_steps = np.linspace(0, width, mosaic_level + 1, dtype="int")
        y_steps = np.linspace(0, height, mosaic_level + 1, dtype="int")

        for i in range(1, len(y_steps)):
            for j in range(1, len(x_steps)):
                start_x = x_steps[j - 1]
                start_y = y_steps[i - 1]
                end_x = x_steps[j]
                end_y = y_steps[i]

                roi = image[start_y:end_y, start_x:end_x]
                (blue, green, red) = [int(x) for x in cv2.mean(roi)[:3]]
                cv2.rectangle(
                    image, (start_x, start_y), (end_x, end_y), (blue, green, red), -1
                )

        return image
