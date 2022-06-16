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
Test for augment undistort node
"""

import cv2
import numpy as np
import pytest
from pathlib import Path

from peekingduck.pipeline.nodes.dabble.camera_calibration import (
    Node,
    _check_corners_validity,
)


class TestCameraCalibration:
    def test_file_io(self):
        with pytest.raises(ValueError) as excinfo:
            Node(
                {
                    "input": ["img"],
                    "output": ["img"],
                    "num_corners": [10, 7],
                    "scale_factor": 4,
                    "file_path": "file.txt",
                }
            )
        assert str(excinfo.value) == "Filepath must have a '.yml' extension."

    def test_check_corner_validity(self, corner_data):
        par_dir = Path(__file__).parent / "camera_calibration" / corner_data
        file_path = str(par_dir)

        data = np.load(file_path)
        width = data["width"]
        height = data["height"]
        corners = data["corners"]
        start_point = data["start_point"]
        end_point = data["end_point"]
        expected_output = data["expected_output"]

        assert (
            _check_corners_validity(width, height, corners, start_point, end_point)
            == expected_output
        )
