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
import pytest

from peekingduck.pipeline.nodes.model.posenetv1.posenet_files.decode_multi import (
    _calculate_keypoint_coords_on_image,
    _change_dimensions,
    _get_instance_score_fast,
    _sort_scored_parts,
    _within_nms_radius_fast,
)

TEST_DIR = Path.cwd() / "images"
NP_FILE = np.load(Path(__file__).resolve().parent / "posenet.npz")


@pytest.fixture
def offsets():
    return NP_FILE["offsets"]


@pytest.fixture
def root_image_coords():
    return NP_FILE["root_image_coords"]


@pytest.fixture
def dst_scores():
    return NP_FILE["dst_scores"]


@pytest.fixture
def dst_keypoints():
    return NP_FILE["dst_keypoints"]


class TestDecodeMulti:
    def test_calculate_keypoint_coords_on_image(self, offsets, root_image_coords):
        root_coords = _calculate_keypoint_coords_on_image(
            heatmap_positions=np.array([4, 6]),
            output_stride=16,
            offsets=offsets,
            keypoint_id=6,
        )
        npt.assert_almost_equal(
            root_coords, root_image_coords, 2
        ), "Incorrect image coordinates"

    def test_within_nms_radius_fast(self, root_image_coords):
        squared_nms_radius = 400
        pose_coords = np.zeros((0, 2))
        check = _within_nms_radius_fast(
            pose_coords, squared_nms_radius, root_image_coords
        )
        assert check == False, "Unable to catch false cases"

        pose_coords = np.array([[65.9072, 99.2803]])
        check = _within_nms_radius_fast(
            pose_coords, squared_nms_radius, root_image_coords
        )
        assert check == True, "Unable to catch true cases"

        pose_coords = np.array([[160.4044, 115.450]])
        check = _within_nms_radius_fast(
            pose_coords, squared_nms_radius, root_image_coords
        )
        assert check == False, "Unable to catch false cases"

    def test_get_instance_score_fast(self, dst_scores, dst_keypoints):
        squared_nms_radius = 400
        keypoint_scores = dst_scores[0]
        keypoint_coords = dst_keypoints[0]
        exist_pose_coords = dst_keypoints[:0, :, :]
        pose_score = _get_instance_score_fast(
            exist_pose_coords, squared_nms_radius, keypoint_scores, keypoint_coords
        )
        npt.assert_almost_equal(
            pose_score, 0.726, 3
        ), "Instance score did not meet expected value"

    def test_change_dimensions(self):
        scores = np.tile(np.array([1, 2, 3]), (2, 2, 1))
        offsets = np.tile(np.array([1, 2]), (2, 2, 3, 1))
        displacements_fwd = displacements_bwd = np.tile(np.array([1, 2]), (2, 2, 2, 1))
        new_offsets, new_dfwd, new_dbwd = _change_dimensions(
            scores, offsets, displacements_fwd, displacements_bwd
        )
        npt.assert_almost_equal(
            new_offsets, np.tile(np.array([[2, 1], [1, 2], [2, 1]]), (2, 2, 1, 1)), 4
        ), "Outputs are incorrect after dimension change"
        npt.assert_almost_equal(
            new_dfwd, np.tile(np.array([[1, 1], [2, 2]]), (2, 2, 1, 1)), 4
        ), "Outputs are incorrect after dimension change"
        npt.assert_almost_equal(
            new_dbwd, np.tile(np.array([[1, 1], [2, 2]]), (2, 2, 1, 1)), 4
        ), "Outputs are incorrect after dimension change"

    def test_sort_scored_parts(self):
        sample_parts = [
            (0.058, 15, np.array([10, 0])),
            (0.924, 12, np.array([5, 11])),
            (0.299, 2, np.array([4, 3])),
            (0.490, 1, np.array([3, 15])),
            (0.806, 0, np.array([15, 12])),
        ]
        expected_parts = [
            (0.924, 12, np.array([5, 11])),
            (0.806, 0, np.array([15, 12])),
            (0.490, 1, np.array([3, 15])),
            (0.299, 2, np.array([4, 3])),
            (0.058, 15, np.array([10, 0])),
        ]
        scored_parts = _sort_scored_parts(sample_parts)
        npt.assert_array_equal(
            np.array(scored_parts, dtype="object")[:, :2],
            np.array(expected_parts, dtype="object")[:, :2],
        ), "Unable to sort scored parts correctly"
