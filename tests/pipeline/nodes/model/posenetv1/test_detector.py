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

from pathlib import Path

import numpy as np
import numpy.testing as npt

from peekingduck.pipeline.nodes.model.posenetv1.posenet_files.detector import (
    _sigmoid,
    get_keypoints_relative_coords,
)

NP_FILE = np.load(Path(__file__).resolve().parent / "posenet.npz")


class TestDetector:
    def test_sigmoid(self):
        x = np.array([[1, 2], [-1, -2]])
        f = _sigmoid(x)
        npt.assert_almost_equal(
            f,
            np.array([[0.731, 0.881], [0.269, 0.119]]),
            3,
            err_msg="Incorrect output after applying sigmoid",
        )

    def test_get_keypoints_relative_coords(self):
        full_keypoint_rel_coords = get_keypoints_relative_coords(
            NP_FILE["full_keypoint_coords"], np.array([2.844, 1.888]), [640, 425]
        )
        assert (
            len(full_keypoint_rel_coords) == 2
        ), "Incorrect number of persons detected"
        assert (
            len(full_keypoint_rel_coords.shape) == 3
        ), "Output keypoint coords should be 3D"
        assert full_keypoint_rel_coords.shape[
            2
        ], "Keypoint coords should be a 2D matrix of 2D offsets"
        npt.assert_almost_equal(
            full_keypoint_rel_coords,
            NP_FILE["full_keypoint_rel_coords"],
            2,
            err_msg="Unexpected output value",
        )
