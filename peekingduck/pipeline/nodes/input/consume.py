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
Consumes a message sent over a network.
"""
import base64
from typing import Any, Dict

import cv2
import numpy as np
from peekingduck.pipeline.nodes.abstract_node import AbstractNode


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """

        img = np.fromstring(
            base64.b64decode(inputs["message"][self.image_key]), np.uint8
        )
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        filename_vals = [
            inputs["message"][filename_key] for filename_key in self.filename_keys
        ]
        filename = "_".join(filename_vals) + "." + self.image_format

        return {
            "img": img,
            "filename": filename,
            "pipeline_end": False,
            "saved_video_fps": 1.0,
        }
