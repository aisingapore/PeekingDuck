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
from __future__ import annotations
from typing import Dict, List, Any
from peekingduck.pipeline.nodes.heuristic.zoningv1.divider import DividerZone
from peekingduck.pipeline.nodes.heuristic.zoningv1.area import Area
from peekingduck.pipeline.nodes.heuristic.zoningv1.zone import Zone
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """Node that checks if any objects are near to each other"""
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)
        zones_info = config["zones"]
        self.zones = [self._create_zone(zone) for zone in zones_info]

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Compares the 3D locations of all objects to see which objects are close to each other.
        If an object is close to another, tag it.

        Args:
            inputs (dict): Dict with keys "obj_3D_locs".

        Returns:
            outputs (dict): Dict with keys "obj_tags".
        """
        num_of_zones = len(self.zones)
        zone_counts = [0] * num_of_zones

        # for each x, y point, check if it is in any zone and add count
        for point in inputs["btm_midpoint"]:
            for i, zone in enumerate(self.zones):
                if zone.point_within_zone(*point):
                    zone_counts[i] += 1

        return {"zones": self.zones,
                "zone_count": zone_counts}

    @staticmethod
    def _create_zone(zone: List[Any]) -> Zone:
        # creates the appropriate Zone class for use zoning analytics
        if zone[0] == "dividers":
            return DividerZone(zone[1])
        if zone[0] == "area":
            return Area(zone[1])
        # if neither, something is wrong. Raise error
        raise TypeError("Zone Type Error: %s is not a type of zone." % zone[0])
