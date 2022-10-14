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

"""Superimposes a heat map over an image."""

from typing import Any, Dict

import cv2
import numpy as np

from peekingduck.pipeline.nodes.abstract_node import AbstractNode


class Node(AbstractNode):  # pylint: disable=too-few-public-methods
    """Superimposes a heat map over an image.

    The :mod:`draw.heat_map` node helps to identify areas that are more
    crowded. Areas that are more crowded are highlighted in red while areas
    that are less crowded are highlighted in blue.

    Inputs:
        |img_data|

        |density_map_data|
        This is produced by nodes such as :mod:`model.csrnet`.

    Outputs:
        |img_data|

    Configs:
        None.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        heat_map_img = self._add_heat_map(inputs["density_map"], inputs["img"])
        outputs = {"img": heat_map_img}
        return outputs

    def _add_heat_map(self, density_map: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Superimposes a heat map over an ``image``.

        Args:
            density_map (np.ndarray): predicted density map.
            image (np.ndarray): image in numpy array.

        Returns:
            image (np.ndarray): image with a heat map superimposed over it.
        """
        if np.count_nonzero(density_map) != 0:
            density_map = self._norm_min_max(density_map)
            heat_map = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)
            image = cv2.addWeighted(image, 0.5, heat_map, 0.5, 0)

        return image

    @staticmethod
    def _norm_min_max(src: np.ndarray) -> np.ndarray:
        target = None
        norm_results = cv2.normalize(
            # source array
            src,
            # destination array
            target,
            # lower boundary value
            alpha=0,
            # upper boundary value
            beta=255,
            # normalization type
            norm_type=cv2.NORM_MINMAX,
            # data type
            dtype=cv2.CV_8U,
        )
        return norm_results
