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
Adjusts brightness of an incoming image
"""


from typing import Any, Dict
import cv2
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """Node for changing image contrast and brightness

    The draw image processor node adjusts contrast and brightness of the given image.
    Uses alogrithm by OpenCV. An article providing a good overview of the algorithm
    can be found `here <https://programmer.ink/think/
    adjusting-the-brightness-and-contrast-of-an-image-with-opencv4.3.0-tutorial
    .html#3、API-convertScaleAbs>`_.

    Inputs:

        |img|

    Outputs:
        |img|

    Configs:
        brightness (:obj:`int`): **[-100,100], default = 0**
            Adjusts the brightness of the image.

        contrast (:obj:`float`): **[1,3], default = 1**
            Adjusts the contrast of the image.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """function that draws pose details onto input image

        Args:
            inputs (dict): Dict with keys "img"
        """
        img = cv2.convertScaleAbs(inputs['img'],
                                  alpha=self.contrast,
                                  beta=self.brightness)
        return {"img": img}
