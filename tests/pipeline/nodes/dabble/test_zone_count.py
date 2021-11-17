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
Test for zone count node
"""

import pytest

from peekingduck.pipeline.nodes.dabble.zone_count import Node


@pytest.fixture
def zone_count():
    node = Node(
        {
            "input": ["btm_midpoint"],
            "output": ["zones", "zone_count"],
            "resolution": [1280, 720],  # Used only in fraction mode
            "zones": [
                [[0, 0], [640, 0], [640, 720], [0, 720]],
                [[0.5, 0], [1, 0], [1, 1], [0.5, 1]],
            ],
        }
    )
    return node


class TestBboxCount:
    def test_no_counts(self, zone_count):
        input1 = {"btm_midpoint": []}
        expected_zones = [
            [(0, 0), (640, 0), (640, 720), (0, 720)],
            [(640, 0), (1280, 0), (1280, 720), (640, 720)],
        ]
        results = zone_count.run(input1)
        counts = results["zone_count"]
        zones = results["zones"]

        assert len(zones) == 2
        assert zones == expected_zones

        assert len(counts) == 2
        assert counts[0] == 0
        assert counts[1] == 0

    def test_counts_in_one_zone(self, zone_count):
        pts = [(2, 2), (3, 3), (4, 4), (639, 0)]
        input1 = {"btm_midpoint": pts}
        results = zone_count.run(input1)["zone_count"]

        assert len(results) == 2
        assert results[0] == 4

    def test_counts_multiple_zones(self, zone_count):
        pts = [(2, 2), (3, 3), (720, 700), (650, 50)]
        input1 = {"btm_midpoint": pts}
        results = zone_count.run(input1)["zone_count"]

        assert len(results) == 2
        assert results[0] == 2
        assert results[1] == 2
