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

from peekingduck.pipeline.nodes.augment.undistort import Node


@pytest.fixture
def undistort():
    par_dir = Path(__file__).parent / "undistort" / "camera_calibration_coeffs.yml"
    file_path = str(par_dir)

    node = Node({"input": ["img"], "output": ["img"], "file_path": file_path})
    return node


class TestUndistort:
    def test_undistort(self, undistort, undistort_before):
        before_img = cv2.imread(undistort_before)

        outputs = undistort.run({"img": before_img})
        assert before_img.shape != outputs["img"].shape

    def test_file_io(self):
        with pytest.raises(ValueError) as excinfo:
            Node({"input": ["img"], "output": ["img"], "file_path": "file.txt"})
        assert str(excinfo.value) == "Filepath must have a '.yml' extension."

        par_dir = Path(__file__).parent / "undistort" / "nonexistent_file.yml"
        file_path = str(par_dir)

        with pytest.raises(FileNotFoundError) as excinfo:
            Node({"input": ["img"], "output": ["img"], "file_path": file_path})
        assert (
            str(excinfo.value)
            == f"File {file_path} does not exist. Please run the camera calibration again."
        )
