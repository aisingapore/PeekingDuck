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

from peekingduck.nodes.abstract_node import AbstractNode
from peekingduck.nodes.base import ThresholdCheckerMixin
from peekingduck.nodes.draw.utils.constants import COLOR_MAP
from peekingduck.nodes.draw.utils.general import get_color
from peekingduck.nodes.draw.utils.pose import Pose


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
            Color of the keypoint connections should either be a string in :ref:`color-palette`,
            or a list of BGR values.

        keypoint_dot_radius (:obj:`int`): **default = 5** |br|
            Radius of the keypoints.

    .. _color-palette:

    .. rubric:: PeekingDuck's Color Palette

    :ref:`color-palette` offers a wide range of default colors [1]_ for the user to choose from.

    .. include:: /include/color_palette.rst

    .. container:: toggle

       .. container:: header

          **Show/Hide color palette**

       |color_palette|

       .. rubric:: Footnotes

       .. [1] Colors with asterisk indicates PeekingDuck's in-house colors.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self._validate_configs()

        self.pose = Pose(
            get_color(self.config["keypoint_dot_color"]),
            get_color(self.config["keypoint_connect_color"]),
            self.keypoint_dot_radius,
        )

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
        valid_color_names = set(COLOR_MAP.keys())
        valid_color_range = "[0, 255]"

        if isinstance(self.config[key], str):
            try:
                self.check_valid_choice(key, valid_color_names)  # type: ignore
            except Exception as wrong_color_choice:
                raise ValueError(
                    f"{key} must be one of {sorted(valid_color_names)} or passed as a \
                    list of BGR values."
                ) from wrong_color_choice
        else:
            self._check_valid_bgr_value(key, valid_color_range)

    def _check_valid_bgr_value(self, key: str, color_range: str) -> None:
        """Checks that configuration value specified by `key` is a list of length 3,
        and if it is, checks that each value is within the range specified by
        color_range.

        Example:
            >>> keypoint_dot_color = [100, 100, 100, 100]
            >>> self._check_valid_bgr_value("keypoint_dot_color", "[0, 255]")

            This will raise a ValueError as the list is of length 4.

            >>> keypoint_dot_color = [100, 100, 300]
            >>> self._check_valid_bgr_value("keypoint_dot_color", "[0, 255]")

            This will raise a ValueError as 300 is out of range.

        Raises:
            ValueError: If the configuration value is not a list of length 3
                or if the BGR values are out of range.
        """
        if len(self.config[key]) != 3:
            raise ValueError("BGR values must be a list of length 3.")
        self.check_bounds(key, color_range)

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {
            "keypoint_dot_color": Union[List[int], str],
            "keypoint_connect_color": Union[List[int], str],
            "keypoint_dot_radius": int,
        }

    def _validate_configs(self) -> None:
        """Validates the config values."""
        self._check_valid_color("keypoint_dot_color")
        self._check_valid_color("keypoint_connect_color")
        self.check_bounds("keypoint_dot_radius", "[0, +inf)")
