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
Estimates the 3D coordinates of an object given a 2D bounding box
"""

from typing import Dict, Any
import numpy as np
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """Node that uses 2D bounding boxes information to estimate 3D location.

    Inputs:
        |bboxes|

    Outputs:
        |obj_3D_locs|

    Configs:
        focal_length (:obj:`float`): **default = 1.14**

            Approximate focal length of webcam used, in metres. Example on measuring focal length:
            https://learnopencv.com/approximate-focal-length-for-webcams-and-cell-phone-cameras/


        height_factor (:obj:`float`): **default = 2.5**

            A factor used to estimate real-world distance from pixels, based on average human height
            in metres. The value varies across different camera set-ups, and calibration may be
            required.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Converts 2D bounding boxes into 3D locations.
        """

        locations = []

        for bbox in inputs["bboxes"]:
            # Subtraction is to make the camera the origin of the coordinate system
            center_2d = ((bbox[0:2] + bbox[2:4]) * 0.5) - np.array([0.5, 0.5])
            bbox_height = bbox[3] - bbox[1]

            z_coord = (self.focal_length * self.height_factor) / bbox_height
            x_coord = (center_2d[0] * self.height_factor) / bbox_height
            y_coord = (center_2d[1] * self.height_factor) / bbox_height

            point = np.array([x_coord, y_coord, z_coord])
            locations.append(point)

        outputs = {"obj_3D_locs": locations}

        return outputs
