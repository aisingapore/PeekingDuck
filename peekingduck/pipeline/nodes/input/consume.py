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
    """Converts an incoming :term:`message` from a network into useful :term:`img` and
    :term:`filename` data types to be used by the PeekingDuck pipeline. This node is designed
    to be the first node in the pipeline and can be used in conjunction with the PeekingDuck
    Server feature. This is an example of what a received :term:`message` could look like: ::

        {
            "frame": /9j/4QAYRXhpZgAASUkqAAg...,
            "time_stamp": 220819-153115-434486,
            "location": XYZ,
            "camera": ABC,
            "level": 3
        }

    The :term:`message` dictionary has to contain 2 key components:

        * An image, encoded in base64 format. In the above example, it is the value of the
          ``frame`` key. This node will use the ``base64.b64decode()`` operation to decode the
          image and perform additional steps to convert it to the :term:`img` format, which can be
          used by other PeekingDuck nodes.
        * Metadata to generate a filename for the resulting image. In the above example, if the
          ``time_stamp`` and ``location`` keys are chosen, the filename is formed by the values of
          these chosen keys separated by underscores, e.g. `220819-153115-434486_XYZ`.

    Inputs:
        |message_data|

    Outputs:
        |img_data|

        |filename_data|

        |pipeline_end_data|

        |saved_video_fps_data|

    Configs:
        image_key (:obj:`str`): **default = "image"** |br|
            Key of :term:`message` to be converted into :term:`img`. See example above for more
            details.
        filename_keys (:obj:`List[str]`): **default = []** |br|
            List of keys in :term:`message` used to form :term:`filename`. The values will be
            separated by underscores - see example above.
        filename_ext (:obj:`str`): **{"jpg", "jpeg", "png"}, default = "jpg"** |br|
            Filename extension of the image to be saved.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.image_ext = ["jpeg", "jpg", "png"]
        self._check_image_ext(self.filename_ext)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """Consumes a message sent over a network."""

        img = np.fromstring(
            base64.b64decode(inputs["message"][self.image_key]), np.uint8
        )
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        filename_vals = [
            inputs["message"][filename_key] for filename_key in self.filename_keys
        ]
        filename = "_".join(filename_vals) + "." + self.filename_ext

        return {
            "img": img,
            "filename": filename,
            "pipeline_end": False,
            "saved_video_fps": 1.0,
        }

    def _check_image_ext(self, ext: str) -> None:
        if ext not in self.image_ext:
            raise ValueError(
                f"{ext} is an unsupported image extension. Only {self.image_ext} extensions"
                " are supported."
            )
