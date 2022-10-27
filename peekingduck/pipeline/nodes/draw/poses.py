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
Draws keypoints on a detected pose.
"""

from typing import Any, Dict, List, Union

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.base import ThresholdCheckerMixin
from peekingduck.pipeline.nodes.draw.utils.constants import COLOR_MAP
from peekingduck.pipeline.nodes.draw.utils.general import get_color
from peekingduck.pipeline.nodes.draw.utils.pose import Pose


class Node(ThresholdCheckerMixin, AbstractNode):
    """Draws poses onto image.

    The :mod:`draw.poses` node uses the :term:`keypoints`,
    :term:`keypoint_scores`, and :term:`keypoint_conns` predictions from pose
    models to draw the human poses onto the image. For better understanding,
    check out the pose models such as :mod:`HRNet <model.hrnet>` and
    :mod:`PoseNet <model.posenet>`.

    Inputs:
        |img_data|

        |keypoints_data|

        |keypoint_scores_data|

        |keypoint_conns_data|

    Outputs:
        |none_output_data|

    Configs:
        keypoint_dot_color (:obj:`Union[List[int], str]`): **default = "tomato"** |br|
            Color of the keypoints should either be a string in :ref:`color-palette`,
            or a list of BGR values.

        keypoint_connect_color (:obj:`Union[List[int], str]`): **default = "champagne"** |br|
            Color of the keypoints should either be a string in :ref:`color-palette`,
            or a list of BGR values.

        keypoint_dot_radius (:obj:`int`): **default = 5** |br|
            Radius of the keypoints.

    **Color Palette**

    :ref:`color-palette` offers a wide range of default colors for the user to choose from.

    .. _color-palette:

    .. list-table:: PeekingDuck's Color Palette
       :widths: 20 20
       :header-rows: 1

       * - Color Palette by Type
         - Color Palette by Name
       * - .. figure:: ../assets/api/color_map_by_type.png

           PeekingDuck's Color Palette Sorted by Color Types [1]_.

         - .. figure:: ../assets/api/color_map_by_name.png

           PeekingDuck's Color Palette Sorted by Color Names [1]_.

    .. rubric:: Footnotes

    .. [1] Colors with asterisk indicates PeekingDuck's in-house colors.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self._validate_configs()

        self.keypoint_dot_color = get_color(self.config["keypoint_dot_color"])
        self.keypoint_connect_color = get_color(self.config["keypoint_connect_color"])

        self.pose = Pose(
            self.keypoint_dot_color,
            self.keypoint_connect_color,
            self.keypoint_dot_radius,
        )

    def _validate_configs(self) -> None:
        """Validates the config values."""
        self._check_valid_color("keypoint_dot_color")
        self._check_valid_color("keypoint_connect_color")
        self.check_bounds("keypoint_dot_radius", "[0, +inf)")

    def _check_valid_color(self, key: str) -> None:
        """Checks that configuration value specified by `key` is a valid color.

        Example:
            >>> keypoint_dot_color = "blueee"
            >>> self._check_valid_color("keypoint_dot_color")

            This will raise a ValueError as "blueee" is not a valid color.

            >>> keypoint_dot_color = [100, 100, 300]
            >>> self._check_valid_color("keypoint_dot_color")

            This will raise a ValueError as 300 is out of range.

        Args:
            key (str): The specified key.

        Raises:
            ValueError: If the configuration value is not either a valid color
                in :ref:`color-palette`, or a list of BGR values, or if the BGR
                values are out of range [0, 255].
        """
        valid_colors_type = set(COLOR_MAP.keys())
        valid_colors_range = "[0, 255]"

        if isinstance(self.config[key], str):
            try:
                self.check_valid_choice(key, valid_colors_type)
            except Exception as wrong_color_choice:
                raise ValueError(
                    f"{key} must be one of {valid_colors_type} or passed as a list of BGR values."
                ) from wrong_color_choice
        else:
            self.check_bounds(key, valid_colors_range)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Draws pose details onto input image.

        Args:
            inputs (dict): Dictionary with keys "img", "keypoints", and
                "keypoint_conns".
        """
        self.pose.draw_human_poses(
            inputs["img"],
            inputs["keypoints"],
            inputs["keypoint_conns"],
        )
        return {}

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {
            "keypoint_dot_color": Union[List[int], str],
            "keypoint_connect_color": Union[List[int], str],
            "keypoint_dot_radius": int,
        }
