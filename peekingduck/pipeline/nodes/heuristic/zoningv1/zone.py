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
from abc import ABCMeta, abstractmethod
from typing import Any


class Zone(metaclass=ABCMeta):
    """An abstract class for zone methods used in zone counting
    """

    def __init__(self, zoning_type: str) -> None:
        self.zoning_type = zoning_type

    @classmethod
    def __subclasshook__(cls: Any, subclass: Any) -> bool:
        return (hasattr(subclass, 'point_within_zone') and
                callable(subclass.point_within_zone) and
                hasattr(subclass, '_is_inside'))

    @abstractmethod
    def point_within_zone(self, x_coord: float, y_coord: float) -> bool:
        """Abstract method. Used to find whether any point is inside a zone.
        """
        return self._is_inside(x_coord, y_coord)

    @abstractmethod
    def _is_inside(self, x_coord: float, y_coord: float) -> bool:
        """Abstact method. Implementation of point check for the zone.
        """
