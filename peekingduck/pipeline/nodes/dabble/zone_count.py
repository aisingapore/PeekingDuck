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

"""Counts the number of detected objects within a boundary."""

from typing import Any, Dict, List, Union

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.dabble.zoningv1.zone import Zone


class Node(AbstractNode):
    """Uses the bottom midpoints of all detected bounding boxes and outputs the
    number of object counts in each specified zone.

    Given the bottom mid-points of all detected objects, this node checks if
    the points fall within the area of the specified zones. The zone counting
    detections depend on the configuration set in the object detection models,
    such as the type of object to detect.

    Inputs:
        |btm_midpoint_data|

    Outputs:
        |zones_data|

        |zone_count_data|

    Configs:
        resolution (:obj:`List[int]`): **default = [1280, 720]**. |br|
            Resolution of input array to calculate pixel coordinates of zone
            points.
        zones (:obj:`List[List[List[Union[int, float]]]]`): |br|
            **default = [**                                     |br|
            |tab| **[[0, 0], [640, 0], [640, 720], [0, 720]],** |br|
            |tab| **[[0.5, 0], [1, 0], [1, 1], [0.5, 1]]**      |br|
            **]**                                               |br|
            Used for creation of specific zones with either the absolute pixel
            values or % of resolution as a fraction between [0, 1].
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.zones = [self._create_zone(zone) for zone in self.zones]  # type: ignore

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Counts all detected objects that falls within any specified zone,
        and return the total object count in each zone.
        """
        zone_counts = [0] * len(self.zones)

        # for each x, y point, check if it is in any zone and add count
        for point in inputs["btm_midpoint"]:
            for i, zone in enumerate(self.zones):
                if zone.contains(point):
                    zone_counts[i] += 1

        return {
            "zones": [zone.polygon_points for zone in self.zones],
            "zone_count": zone_counts,
        }

    def _create_zone(self, zone: List[List[Union[float, int]]]) -> Zone:
        """Creates the appropriate Zone given either the absolute pixel values
        or % of resolution as a fraction between [0, 1].
        """
        if all(all(0 <= i <= 1 for i in coords) for coords in zone):
            # coordinates are in fraction. Use resolution to get correct coords
            zone_points = [
                self._get_pixel_coords(coords, self.resolution) for coords in zone
            ]
        elif all(
            all((isinstance(i, int) and i >= 0) for i in coords) for coords in zone
        ):
            # list is in pixel value.
            zone_points = zone  # type: ignore
        else:
            raise ValueError(
                f"Zone {zone} needs to be all pixel-wise points or all fractions "
                "of the frame between 0 and 1. Please check zone_count configs."
            )
        created_zone = Zone(zone_points)

        return created_zone

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {"resolution": List[int], "zones": List[List[List[Union[int, float]]]]}

    @staticmethod
    def _get_pixel_coords(
        coords: List[Union[float, int]], resolution: List[int]
    ) -> List[int]:
        """Returns the pixel position of the zone points."""
        return [int(coords[0] * resolution[0]), int(coords[1] * resolution[1])]
