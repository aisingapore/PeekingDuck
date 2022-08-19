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
Blurs area bounded by bounding boxes over detected object.
"""

from typing import Any, Dict, List

import cv2
import numpy as np

from peekingduck.pipeline.nodes.abstract_node import AbstractNode


class Node(AbstractNode):  # pylint: disable=too-few-public-methods
    """Blurs area bounded by bounding boxes on image.

    The :mod:`draw.blur_bbox` node blurs the areas of the image bounded by the
    bounding boxes output from an object detection model.

    Inputs:
        |img_data|

        |bboxes_data|

    Outputs:
        |img_data|

    Configs:
        blur_kernel_size (:obj:`int`): **default = 50**. |br|
            This defines the kernel size used in the blur filter. Larger values
            of ``blur_kernel_size`` gives more intense blurring.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Reads the image input and returns the image, with the areas bounded
        by the bboxes blurred.

        Args:
            inputs (dict): Dictionary of inputs with keys "img", "bboxes"

        Returns:
            outputs (dict): Output in dictionary format with key "img"
        """
        blurred_img = self._blur_bbox(inputs["img"], inputs["bboxes"])
        outputs = {"img": blurred_img}
        return outputs

    def _blur_bbox(self, image: np.ndarray, bboxes: List[np.ndarray]) -> np.ndarray:
        """Blurs areas bounded by bounding boxes on ``image``.

        Args:
            image (np.ndarray): Image in numpy array.
            bboxes (List[np.ndarray]): numpy array of detected bboxes

        Returns:
            (np.ndarray): Image with blurred bounding box regions.
        """
        height = image.shape[0]
        width = image.shape[1]

        for bbox in bboxes:
            x_1, y_1, x_2, y_2 = bbox
            y_1, y_2 = int(y_1 * height), int(y_2 * height)
            x_1, x_2 = int(x_1 * width), int(x_2 * width)

            # to get the area bounded by bbox
            bbox_image = image[y_1:y_2, x_1:x_2, :]

            # apply the blur using blur filter from opencv
            blur_bbox_image = cv2.blur(
                bbox_image, (self.blur_kernel_size, self.blur_kernel_size)
            )
            image[y_1:y_2, x_1:x_2, :] = blur_bbox_image

        return image

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {"blue_kernel_size": int}
