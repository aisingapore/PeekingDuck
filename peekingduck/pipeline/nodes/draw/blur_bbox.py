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
from peekingduck.pipeline.nodes.node import AbstractNode
from scipy.ndimage import gaussian_filter


class Node(AbstractNode):
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

    def blur(self, bboxes, image):
        """
        Function that blur the area bounded by bbox in an image
        """
        frameHeight = image.shape[0]
        frameWidth = image.shape[1]

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            x1, x2 = int(x1 * frameWidth), int(x2 * frameWidth)
            y1, y2 = int(y1 * frameHeight), int(y2 * frameHeight)

            # slice the image to get the area bounded by bbox
            bbox_image = image[y1:y2, x1:x2, :].copy()
            # apply the blur using gaussian filter from scipy
            blur_bbox_image = gaussian_filter(bbox_image, sigma=5)
            image[y1:y2, x1:x2, :] = blur_bbox_image

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
