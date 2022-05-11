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
Adjusts the contrast of an image.
"""


from typing import Any, Dict

import cv2

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.base import ThresholdCheckerMixin


class Node(ThresholdCheckerMixin, AbstractNode):
    """Adjusts the contrast of an image, by multiplying with a gain/`alpha
    parameter <https://docs.opencv.org/4.x/d3/dc1/tutorial_basic_
    linear_transform.html>`_.

    Inputs:
        |img_data|

    Outputs:
        |img_data|

    Configs:
        alpha (:obj:`float`): **[0, 3], default = 1**. |br|
            Increasing the value of alpha increases the contrast.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.check_bounds("alpha", "[0, 3]")

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Adjusts the contrast of an image frame.

        Args:
            inputs (Dict): Inputs dictionary with the key `img`.

        Returns:
            (Dict): Outputs dictionary with the key `img`.
        """
        img = cv2.convertScaleAbs(inputs["img"], alpha=self.alpha, beta=0)

        return {"img": img}
