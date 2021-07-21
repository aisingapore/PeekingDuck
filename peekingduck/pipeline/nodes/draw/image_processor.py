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

    The draw bbox node uses the bboxes and, optionally, the bbox labels from the model 
    predictions to draw the bbox predictions onto the image. 
    For better understanding of the usecase, refer to the object counting usecase.
    
    Inputs:

        |img|

        |bboxes|

        |bbox_labels|

    Outputs:
        |none|

    Configs:
        brightness (:obj:`int`):
            Adjusts the brightness of the image. Takes integer values -100 to 100.
            Default value is 0 (no change in brightness).

        contrast (:obj:`float`):
            Adjusts the contrast of the image. Takes values 1 to 3.
            Default value is 1 (no change in contrast).
    """

    def __init__(self, config: Dict[str, Any]) -> None:

        super().__init__(config, node_path=__name__)
        self.brightness = config['brightness']
        self.contrast = config ['contrast']

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """function that draws pose details onto input image

        Args:
            inputs (dict): Dict with keys "img"
        """
        img = cv2.convertScaleAbs(inputs['img'],
                                  alpha=self.contrast,
                                  beta=self.brightness)
        return {"img": img}
