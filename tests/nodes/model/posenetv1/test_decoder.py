# Copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
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
import scipy.ndimage as ndi
import yaml

from peekingduck.nodes.model.posenetv1.posenet_files.constants import OUTPUT_STRIDE
from peekingduck.nodes.model.posenetv1.posenet_files.decoder import Decoder
from tests.conftest import PKD_DIR

DECIMAL_PRECISION = 4
NP_FILE = np.load(Path(__file__).resolve().parent / "posenet.npz")
SOURCE_KEYPOINT = np.array([76.23, 70.98])


@pytest.fixture(name="decoder", params=[True, False])
def fixture_decoder(request):
    """Instantiates Decoder object both `use_jit` values."""
    with open(PKD_DIR / "configs" / "model" / "posenet.yml") as infile:
        node_config = yaml.safe_load(infile)
        node_config["use_jit"] = request.param

        decoder = Decoder(node_config["score_threshold"], node_config["use_jit"])

        return decoder


def test_build_parts_with_peaks(decoder):
    expected_parts = [
        (0.9, 1, np.array([2, 3])),
        (0.8, 6, np.array([7, 8])),
        (0.7, 11, np.array([12, 12])),
    ]
    scores = np.random.random_sample((15, 15, 17)) * 0.1
    for score, keypoint_id, coords in expected_parts:
        scores[coords[0], coords[1], keypoint_id] = score
    scores[1, 12, 13] = 0.4  # add a point which gets filtered out by score
    local_peaks = ndi.maximum_filter(scores, size=(3, 3, 1), mode="constant")

    parts = decoder._build_parts_with_peaks(scores, 0.5, local_peaks)

    assert len(expected_parts) == len(parts)
    for i, _ in enumerate(expected_parts):
        npt.assert_almost_equal(parts[i][0], expected_parts[i][0], DECIMAL_PRECISION)
        assert parts[i][1] == expected_parts[i][1]
        npt.assert_array_equal(parts[i][2], expected_parts[i][2])


def test_point_is_beyond_radius(decoder):
    radius = 12
    existing_coords = np.ones((10, 2)) * SOURCE_KEYPOINT
    # setup mask so some points are move in positive direction while the rest
    # are moved in the negative direction
    mask = np.random.choice([True, False], existing_coords.shape)
    # setup positive offsets ranged [radius, 2 * radius)
    pos_offsets = np.ma.array(
        (np.random.random_sample(existing_coords.shape) + 1) * radius,
        mask=mask,
        fill_value=0,
    )
    # setup negative offsets ranged [-2 * radius, -radius)
    neg_offsets = np.ma.array(
        (np.random.random_sample(existing_coords.shape) - 2) * radius,
        mask=~mask,
        fill_value=0,
    )
    existing_coords += pos_offsets + neg_offsets

    assert not decoder._is_within_nms_radius(
        existing_coords, radius**2, SOURCE_KEYPOINT
    )


def test_point_is_within_radius(decoder):
    radius = 34
    existing_coords = np.ones((10, 2)) * SOURCE_KEYPOINT
    # setup mask so some points are move in positive direction while the rest
    # are moved in the negative direction
    mask = np.random.choice([True, False], existing_coords.shape)
    # setup positive offsets ranged [radius, 2 * radius)
    pos_offsets = np.ma.array(
        (np.random.random_sample(existing_coords.shape) + 1) * radius,
        mask=mask,
        fill_value=0,
    )
    # setup negative offsets ranged [-2 * radius, -radius)
    neg_offsets = np.ma.array(
        (np.random.random_sample(existing_coords.shape) - 2) * radius,
        mask=~mask,
        fill_value=0,
    )
    existing_coords += pos_offsets + neg_offsets

    # pass in a larger radius
    assert decoder._is_within_nms_radius(
        existing_coords, 4 * radius**2, SOURCE_KEYPOINT
    )


def test_traverse_to_keypoint_with_backward_displacements(decoder):
    score, coords = decoder._traverse_to_target_keypoint(
        edge_id=0,
        source_keypoint=SOURCE_KEYPOINT,
        target_keypoint_id=0,
        scores=NP_FILE["scores"],
        offsets=NP_FILE["offsets"],
        output_stride=OUTPUT_STRIDE,
        displacements=NP_FILE["displacements_bwd"],
    )

    npt.assert_almost_equal(score, 0.9706, DECIMAL_PRECISION)
    npt.assert_almost_equal(coords, np.array([74.9044, 72.2198]), DECIMAL_PRECISION)


def test_traverse_to_keypoint_with_forward_displacements(decoder):
    score, coords = decoder._traverse_to_target_keypoint(
        edge_id=1,
        source_keypoint=SOURCE_KEYPOINT,
        target_keypoint_id=3,
        scores=NP_FILE["scores"],
        offsets=NP_FILE["offsets"],
        output_stride=OUTPUT_STRIDE,
        displacements=NP_FILE["displacements_fwd"],
    )
    npt.assert_almost_equal(score, 0.4393, DECIMAL_PRECISION)
    npt.assert_almost_equal(coords, np.array([77.1059, 71.6157]), DECIMAL_PRECISION)
