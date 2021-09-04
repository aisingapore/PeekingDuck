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
Blur area bounded by bounding boxes over detected object
"""

from typing import Any, Dict
from scipy.ndimage import gaussian_filter
from peekingduck.pipeline.nodes.node import AbstractNode

<<<<<<< HEAD
class Node(AbstractNode): # pylint: disable=R0903
=======
# pylint: disable=R0903
class Node(AbstractNode):
>>>>>>> a0c5d9c434f4bdee94b12bc6497d04d9940efb1d
    """Blur area bounded by bounding boxes on image.

    The draw blur_bbox node uses the bboxes and blur the area of the image
    bounded by the bboxes.

    Inputs:

        |img|

        |bboxes|

    Outputs:
        |none|

    Configs:
        None.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    @staticmethod
    def blur(bboxes, image):
        """
        Function that blur the area bounded by bbox in an image
        """
        height = image.shape[0]
        width = image.shape[1]

        for bbox in bboxes:
            x_1, y_1, x_2, y_2 = bbox
            x_1, x_2 = int(x_1 * width), int(x_2 * width)
            y_1, y_2 = int(y_1 * height), int(y_2 * height)

            # slice the image to get the area bounded by bbox
            bbox_image = image[y_1:y_2, x_1:x_2, :].copy()
            # apply the blur using gaussian filter from scipy
            blur_bbox_image = gaussian_filter(bbox_image, sigma=5)
            image[y_1:y_2, x_1:x_2, :] = blur_bbox_image

        return image

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """function that reads the image input and returns the bboxes
        of the specified objects chosen to be detected
        Args:
            inputs (Dict): Dictionary of inputs with keys "img", "bboxes"
        Returns:
            outputs (Dict): Output in dictionary format with key
            "img"
        """
        blurred_img = self.blur(inputs["bboxes"], inputs["img"])
        outputs = {"img": blurred_img}
        return outputs
