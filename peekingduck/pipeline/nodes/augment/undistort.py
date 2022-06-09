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
Undistorts an image.
"""


from typing import Any, Dict

import cv2, yaml
import numpy as np
from pathlib import Path

from peekingduck.pipeline.nodes.abstract_node import AbstractNode

class Node(AbstractNode):
    """Undistorts an image by removing radial distortion
    <https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html>'_.

    Inputs:
        |img_data|

    Outputs:
        |img_data|

    Configs:
        file_path (:obj:`str`):
            **default = "PeekingDuck/data/camera_calibration_coeffs.yml"**. |br|
            Path of the YML file containing calculated camera coefficients.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.file_path = Path(self.file_path) # type: ignore
        # check if file_path has a '.yml' extension
        if self.file_path.suffix != '.yml':
            raise ValueError("Filepath must have a '.yml' extension.")
        if not self.file_path.exists():
            raise FileNotFoundError(f"File {self.file_path} does not exist. Please run the camera calibration again.")
        
        yaml_file = yaml.safe_load(open(self.file_path))
        self.camera_matrix = np.array(yaml_file['camera_matrix'])
        self.distortion_coeffs = np.array(yaml_file['distortion_coeffs'])

        self.new_camera_matrix = None
        self.roi = None

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Undistorts an image by removing radial distortion

        Args:
            inputs (dict): Inputs dictionary with the key `img`.

        Returns:
            outputs (dict): Outputs dictionary with the key `img`.
        """

        img = inputs["img"]

        if self.new_camera_matrix is None or self.roi is None:
            h, w = img.shape[:2]
            self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_coeffs, (w, h), 0, (w, h))

        undistorted_img = cv2.undistort(img, self.camera_matrix, self.distortion_coeffs, None, self.new_camera_matrix)
        x, y, w, h = self.roi
        undistorted_img = undistorted_img[y:y+h, x:x+w]

        return {"img": undistorted_img}

