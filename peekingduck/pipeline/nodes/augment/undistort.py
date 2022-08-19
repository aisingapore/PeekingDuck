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
Removes distortion from a wide-angle camera image.
"""

from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import yaml

from peekingduck.pipeline.nodes.abstract_node import AbstractNode


class Node(AbstractNode):
    """Undistorts an image by removing `radial and tangential distortion
    <https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html>`_.
    This may help to improve the performance of certain models.

    Before using this node for the first time, please follow the tutorial in
    :mod:`dabble.camera_calibration` to calculate the camera coefficients
    of the camera you are using, and ensure that the file_path that the
    coefficients are stored in is the same as the one specified in the configs.

    The images below show an example of an image before and after undistortion.
    Note that after undistortion, the shape of the image will change and the FOV
    will be reduced slightly.

    .. figure:: /assets/api/undistort.png
        :width: 80 %

        Before undistortion (left) and after undistortion (right)

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

        self.file_path = Path(self.file_path)  # type: ignore
        if self.file_path.suffix != ".yml":
            raise ValueError("Filepath must have a '.yml' extension.")
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"File {self.file_path} does not exist. Please run the camera calibration again "
                "with the dabble.camera_calibation node. You may refer to this tutorial: "
                "https://peekingduck.readthedocs.io/en/stable/nodes/dabble.camera_calibration.html"
            )

        yaml_file = yaml.safe_load(open(self.file_path))
        self.camera_matrix = np.array(yaml_file["camera_matrix"])
        self.distortion_coeffs = np.array(yaml_file["distortion_coeffs"])

        self.new_camera_matrix = None

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Undistorts an image by removing radial distortion

        Args:
            inputs (dict): Inputs dictionary with the key `img`.

        Returns:
            outputs (dict): Outputs dictionary with the key `img`.
        """
        img = inputs["img"]

        if self.new_camera_matrix is None:
            height, width = img.shape[:2]
            self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix,
                self.distortion_coeffs,
                (width, height),
                0,
                (width, height),
            )

        undistorted_img = cv2.undistort(
            img,
            self.camera_matrix,
            self.distortion_coeffs,
            None,
            self.new_camera_matrix,
        )

        x_pos, y_pos, img_w, img_h = self.roi
        undistorted_img = undistorted_img[y_pos : y_pos + img_h, x_pos : x_pos + img_w]

        return {"img": undistorted_img}

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {"file_path": str}
