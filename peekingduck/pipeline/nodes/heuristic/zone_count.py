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
Counts the number of detected objects within a boundary
"""

from typing import Dict, List, Any
from peekingduck.pipeline.nodes.heuristic.zoningv1.zone import Zone
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """Node that checks if any objects are near to each other"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)
        zones_info = config["zones"]
        try:
            self.zones = [self._create_zone(zone, config["resolution"])
                          for zone in zones_info]
        except TypeError as error:
            self.logger.warning(error)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Counts all detected objects that falls within any specified zone,
        and return the total object count in each zone.

        Args:
            inputs (dict): Dict with keys "btm_midpoints".

        Returns:
            outputs (dict): Dict with keys "zones" for tuple of (x, y) points that form zone,
            and "zone_count" for the zone counting of all objects detected in the zone.
        """
        num_of_zones = len(self.zones)
        zone_counts = [0] * num_of_zones

        # for each x, y point, check if it is in any zone and add count
        for point in inputs["btm_midpoint"]:
            for i, zone in enumerate(self.zones):
                if zone.point_within_zone(*point):
                    zone_counts[i] += 1

        return {"zones": [zone.get_all_points_of_area() for zone in self.zones],
                "zone_count": zone_counts}

    def _create_zone(self, zone: List[Any], resolution: List[int]) -> Any:
        """creates the appropriate Zone given either the absolute pixel values or
        % of resolution as a fraction between [0, 1]"""
        created_zone = None

        if all(all(0 <= i <= 1 for i in coords) for coords in zone):
            # coordinates are in fraction. Use resolution to get correct coords
            pixel_coords = [self._get_pixel_coords(coords, resolution) for coords in zone]
            created_zone = Zone(pixel_coords)
        if all(all((isinstance(i, int) and i >= 0) for i in coords) for coords in zone):
            # when 1st-if fails and this statement passes, list is in pixel value.
            created_zone = Zone(zone)

        # if neither, something is wrong
        if not created_zone:
            assert False, ("Zone %s needs to be all pixel-wise points or "
                           "all fractions of the frame between 0 and 1. "
                           "please check zone_count configs." % zone)

        return created_zone

    @staticmethod
    def _get_pixel_coords(coords: List[float], resolution: List[int]) -> List[float]:
        """returns the pixel position of the zone points"""
        return [int(coords[0] * resolution[0]), int(coords[1] * resolution[1])]
