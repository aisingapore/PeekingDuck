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

from peekingduck.nodes.model.posenetv1.posenet_files.decode_multi import (
    _calculate_keypoint_coords_on_image,
    _change_dimensions,
    _get_instance_score_fast,
    _is_within_nms_radius,
)

NP_FILE = np.load(Path(__file__).resolve().parent / "posenet.npz")


class TestDecodeMulti:
    def test_calculate_keypoint_coords_on_image(self):
        root_coords = _calculate_keypoint_coords_on_image(
            heatmap_positions=np.array([4, 6]),
            output_stride=16,
            offsets=NP_FILE["offsets"],
            keypoint_id=6,
        )
        npt.assert_almost_equal(
            root_coords,
            NP_FILE["root_image_coords"],
            2,
            err_msg="Incorrect image coordinates",
        )

    def test_is_within_nms_radius(self):
        squared_nms_radius = 400
        pose_coords = np.zeros((0, 2))
        check = _is_within_nms_radius(
            pose_coords, squared_nms_radius, NP_FILE["root_image_coords"]
        )
        assert not check, "Unable to catch false cases"

        pose_coords = np.array([[65.9072, 99.2803]])
        check = _is_within_nms_radius(
            pose_coords, squared_nms_radius, NP_FILE["root_image_coords"]
        )
        assert check, "Unable to catch true cases"

        pose_coords = np.array([[160.4044, 115.450]])
        check = _is_within_nms_radius(
            pose_coords, squared_nms_radius, NP_FILE["root_image_coords"]
        )
        assert not check, "Unable to catch false cases"

    def test_get_instance_score_fast(self):
        squared_nms_radius = 400
        keypoint_scores = NP_FILE["dst_scores"][0]
        keypoint_coords = NP_FILE["dst_keypoints"][0]
        exist_pose_coords = NP_FILE["dst_keypoints"][:0, :, :]
        pose_score = _get_instance_score_fast(
            exist_pose_coords, squared_nms_radius, keypoint_scores, keypoint_coords
        )
        npt.assert_almost_equal(
            pose_score, 0.726, 3, err_msg="Instance score did not meet expected value"
        )

    def test_change_dimensions(self):
        offsets = np.tile(np.array([1, 2]), (2, 2, 3, 1))
        displacements_fwd = displacements_bwd = np.tile(np.array([1, 2]), (2, 2, 2, 1))
        new_offsets, new_dfwd, new_dbwd = _change_dimensions(
            offsets, displacements_fwd, displacements_bwd
        )
        npt.assert_almost_equal(
            new_offsets,
            np.tile(np.array([[2, 1], [1, 2], [2, 1]]), (2, 2, 1, 1)),
            4,
            err_msg="Outputs are incorrect after dimension change",
        )
        npt.assert_almost_equal(
            new_dfwd,
            np.tile(np.array([[1, 1], [2, 2]]), (2, 2, 1, 1)),
            4,
            err_msg="Outputs are incorrect after dimension change",
        )
        npt.assert_almost_equal(
            new_dbwd,
            np.tile(np.array([[1, 1], [2, 2]]), (2, 2, 1, 1)),
            4,
            err_msg="Outputs are incorrect after dimension change",
        )
