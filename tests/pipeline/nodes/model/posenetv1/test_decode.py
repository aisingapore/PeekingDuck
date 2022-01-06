"""
Copyright 2021 AI Singapore

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from peekingduck.pipeline.nodes.model.posenetv1.posenet_files.decode import (
    _clip_to_indices,
    _traverse_to_target_keypoint,
)

TEST_DIR = Path.cwd() / "images"
NP_FILE = np.load(Path(__file__).resolve().parent / "posenet.npz")


@pytest.fixture
def scores():
    return NP_FILE["scores"]


@pytest.fixture
def offsets():
    return NP_FILE["offsets"]


@pytest.fixture
def displacements_fwd():
    return NP_FILE["displacements_fwd"]


@pytest.fixture
def displacements_bwd():
    return NP_FILE["displacements_bwd"]


@pytest.fixture
def source_keypoint():
    return np.array([76.23, 70.98])


class TestDecode:
    def test_clip_to_indices(self, source_keypoint):
        source_keypoint_indices = _clip_to_indices(
            keypoints=source_keypoint, output_stride=16, width=14, height=14
        )
        npt.assert_almost_equal(
            source_keypoint_indices, np.array([5, 4]), 4
        ), "Unexpected output from clipping to indices"

    def test_traverse_to_target_keypoint(
        self, source_keypoint, scores, offsets, displacements_bwd, displacements_fwd
    ):
        score, coords = _traverse_to_target_keypoint(
            edge_id=0,
            source_keypoint=source_keypoint,
            target_keypoint_id=0,
            scores=scores,
            offsets=offsets,
            output_stride=16,
            displacements=displacements_bwd,
        )
        assert score == pytest.approx(0.9706, 0.01), "Score did not meet expected value"
        npt.assert_almost_equal(score, 0.9706, 4), "Score did not meet expected value"
        npt.assert_almost_equal(
            coords, np.array([74.90, 72.22]), 2
        ), "Coordinates of incorrect values"

        score, coords = _traverse_to_target_keypoint(
            edge_id=1,
            source_keypoint=np.array([76.23, 70.98]),
            target_keypoint_id=3,
            scores=scores,
            offsets=offsets,
            output_stride=16,
            displacements=displacements_fwd,
        )
        npt.assert_almost_equal(score, 0.4393, 4), "Score did not meet expected value"
        npt.assert_almost_equal(
            coords, np.array([77.11, 71.62]), 2
        ), "Coordinates of incorrect values"
