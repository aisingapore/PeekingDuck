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

"""
Algorithms that perform calculations/heuristics on the outputs of ``model``.
"""

import sys
from typing import TYPE_CHECKING

from peekingduck.utils.lazy_module import ImportStructure, _LazyModule

_import_structure: ImportStructure = {
    "bbox_count": [],
    "bbox_to_3d_loc": [],
    "bbox_to_btm_midpoint": [],
    "camera_calibration": [],
    "check_large_groups": [],
    "check_nearby_objs": [],
    "fps": [],
    "group_nearby_objs": [],
    "keypoints_to_3d_loc": [],
    "statistics": [],
    "tracking": [],
    "zone_count": [],
}

if TYPE_CHECKING:
    from . import (
        bbox_count,
        bbox_to_3d_loc,
        bbox_to_btm_midpoint,
        camera_calibration,
        check_large_groups,
        check_nearby_objs,
        fps,
        group_nearby_objs,
        keypoints_to_3d_loc,
        statistics,
        tracking,
        zone_count,
    )
else:
    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, __spec__
    )
