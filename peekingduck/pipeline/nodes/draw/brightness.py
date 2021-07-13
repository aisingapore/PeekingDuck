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
from PIL import Image, ImageEnhance
import numpy as np
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """Node for changing image brightness"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)
        self.brightness = config['brightness']

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """function that draws pose details onto input image

        Args:
            inputs (dict): Dict with keys "img"

        Returns:
            outputs (Dict): bbox output in dictionary format with keys
            "img"
        """
        img = Image.fromarray(inputs["img"])
        enhancer = ImageEnhance.Brightness(img)
        enhanced_img = enhancer.enhance(self.brightness)
        inputs['img'] = np.array(enhanced_img)

        return {"img": np.array(enhanced_img)}
