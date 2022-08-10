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
Draws instance segmentation masks.
"""

import colorsys
from random import randint
from pydoc import locate
from typing import Any, Callable, Dict, List, Tuple, cast

import cv2
import numpy as np

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.base import ThresholdCheckerMixin
from peekingduck.pipeline.nodes.draw.utils.constants import (
    SATURATION_STEPS,
    SATURATION_MINIMUM,
    ALPHA,
    CONTOUR_COLOR,
    CLASS_COLORS,
    DEFAULT_CLASS_COLOR,
)


class Node(AbstractNode, ThresholdCheckerMixin):
    # pylint: disable=line-too-long
    """Draws instance segmentation masks on image.

    The :mod:`draw.mask` node draws instance segmentation masks onto the
    detected object instances.

    Inputs:
        |img_data|

        |masks_data|

        |bbox_labels_data|

    Outputs:
        |img_data|

    Configs:
        instance_color_scheme (:obj:`str`): **{"random", "hue_family"},
            default = "hue_family"** |br|
            This defines what colors to use for the standard masks. |br|
            "hue_family": use the same hue for each instance belonging to
            the same class, but with a slightly different saturation. |br|
            "random": use a random color for all instances.
        effect (:obj:`dict`): **{contrast: null, brightness: null,
            gamma_correction: null, blur: null, mosaic: null}** |br|
            This defines the effect (if any) to apply to either the masked
            (objects) or unmasked (background) areas of the image. If no effect
            is selected, a "standard" instance segmentation mask will be drawn
            and colored according to the instance_color_scheme. |br|
            For example, to apply the contrast effect to the objects in the
            image, set the following config in pipeline_config.yml:

                effect : {contrast: 1.2}

            Note that at most one effect can be enabled at a time.

            +------------------+---------------------------------------+-----------+-------------+
            | Effect           | Description                           | Data Type | Range       |
            +------------------+---------------------------------------+-----------+-------------+
            | contrast         | Adjusts contrast using this value     | float     | [0.0, 3.0]  |
            |                  | as the "alpha" parameter.             |           |             |
            +------------------+---------------------------------------+-----------+-------------+
            | brightness       | Adjusts brightness using this         | int       | [-100, 100] |
            |                  | value as the "beta" parameter.        |           |             |
            +------------------+---------------------------------------+-----------+-------------+
            | gamma_correction | Adjusts gamma using this value as     | float     | [0.0, +inf] |
            |                  | the "gamma" parameter.                |           |             |
            +------------------+---------------------------------------+-----------+-------------+
            | blur             | Blurs the area using this value as    | int       | [1, +inf]   |
            |                  | the "blur_kernel_size" parameter. |br||           |             |
            |                  | Larger values gives more intense      |           |             |
            |                  | blurring.                             |           |             |
            +------------------+---------------------------------------+-----------+-------------+
            | mosaic           | Mosaics the area using this value     | int       | [1, +inf]   |
            |                  | as the resolution of a mosaic         |           |             |
            |                  | filter |br| (width |times| height).   |           |             |
            |                  | The number corresponds to the number  |           |             |
            |                  | of rows |br| and columns used to      |           |             |
            |                  | create a mosaic. For example, the     |           |             |
            |                  | setting |br| (``mosaic: 25``) creates |           |             |
            |                  | a :math:`25 \\times 25` mosaic         |           |             |
            |                  | filter. Increasing |br| the number    |           |             |
            |                  | increases the intensity of            |           |             |
            |                  | pixelation over an area.              |           |             |
            +------------------+---------------------------------------+-----------+-------------+

        effect_area (:obj:`str`): **{"objects", "background"}, default =
            "objects"** |br|
            This defines where the effect should be applied. |br|
            "objects": the effect is applied to the masked areas of the
            image. |br|
            "background": the effect is applied to the unmasked areas of the
            image.
        contours (:obj:`dict`): **{show: False, thickness: 2}** |br|

            +-----------+------------------------------------------------------------+-----------+-----------+
            | Contours  | Description                                                | Data Type | Range     |
            +-----------+------------------------------------------------------------+-----------+-----------+
            | show      | This determines whether to show the contours of the masks. | bool      | N.A.      |
            +-----------+------------------------------------------------------------+-----------+-----------+
            | thickness | This defines the thickness of the contours.                | int       | [1, +inf] |
            +-----------+------------------------------------------------------------+-----------+-----------+
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.class_instance_colors: Dict[str, List[Tuple[int, int, int]]] = {}
        self._validate_configs()

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Reads an input image and returns the image (1) with the instance
        segmentation masks drawn over the detected object instances, or
        (2) with a visual effect mask applied to it. The visual effect can be
        applied to the masked areas of the image or to the unmasked areas.

        Args:
            inputs (dict): Dictionary of inputs with keys "img", "masks",
            "bbox_labels".

        Returns:
            outputs (dict): Output in dictionary format with key "img".
        """
        if self.effect is None:
            output_img = self._draw_standard_masks(
                inputs["img"],
                inputs["masks"],
                inputs["bbox_labels"],
            )
        else:
            output_img = self._mask_apply_effect(
                inputs["img"],
                inputs["masks"],
                self.effect,
            )

        return {"img": output_img}

    def _validate_configs(self) -> None:
        self.check_valid_choice("instance_color_scheme", {"random", "hue_family"})

        effects_count = 0
        self.effect = None
        for effect, setting in self.config["effect"].items():
            if setting:
                effects_count += 1
                self.effect = effect

        if effects_count > 1:
            raise ValueError("A maximum of one effect can be enabled at a time.")

        self.check_valid_choice("effect_area", {"objects", "background"})

        valid_types = [
            "effect|contrast, float",
            "effect|brightness, int",
            "effect|gamma_correction, float",
            "effect|blur, int",
            "effect|mosaic, int",
            "contours|show, bool",
            "contours|thickness, int",
        ]

        valid_ranges = [
            "effect|contrast, [0.0, 3.0]",
            "effect|brightness, [-100, 100]",
            "effect|gamma_correction, [0.0, +inf]",
            "effect|blur, [1, +inf]",
            "effect|mosaic, [1, +inf]",
            "contours|thickness, [1, +inf]",
        ]

        self._check_configs(valid_types, criteria="type")
        self._check_configs(valid_ranges, criteria="range")

    def _check_configs(self, valid_settings: List[str], criteria: str) -> None:
        if criteria == "type":
            self._check_config_values(valid_settings, Node._type_checker)
        elif criteria == "range":
            self._check_config_values(valid_settings, Node._range_checker)
        else:
            raise ValueError("'criteria' parameter must be either 'type' or 'range'.")

    @staticmethod
    def _type_checker(var_name: str, value: Any, var_type: str) -> None:
        """Checks if the value is of the correct type."""
        if value is not None and not isinstance(value, locate(var_type)):  # type: ignore
            raise ValueError(f"{var_name} must be a {var_type} value.")

    @staticmethod
    def _range_checker(var_name: str, value: Any, number_range: str) -> None:
        """Checks if a value is within a range.  The format for the number
        range is "[lower, upper]"."""
        if value is not None:
            lower, upper = [
                float(value.strip()) for value in number_range[1:-1].split(",")
            ]
            if not lower <= value <= upper:
                raise ValueError(
                    f"{var_name} must be within the range of {number_range}."
                )

    def _check_config_values(
        self, valid_settings: List[str], checker: Callable
    ) -> None:
        """Checks the configs for valid data. The checker parameter determines
        the criteria that the config values are checked against. The
        valid_settings can handle nested configs, with names of different
        levels separated by a pipe (|).

        Example to illustrate format of 'valid_settings' parameter:

            config = {
                "not_nested": 10,
                "single_level_dict": {
                    "contrast": 50.0,
                    "text_setting": "sample string",
                },
                "nested_dict": {
                    "effect": {
                        "blur": 50,
                    }
                },
            }

            valid_settings_for_types = [
                "not_nested, int",
                "single_level_dict|contrast, float",
                "single_level_dict|text_setting, str",
                "nested_dict|effect|blur, int",
            ]

            valid_settings_for_ranges = [
                "not_nested, [1, 10]",
                "single_level_dict|contrast, [0, 3]",
                "nested_dict|effect|blur, [1, +inf]",
            ]
        """
        for valid_setting in valid_settings:
            criteria_input_list = valid_setting.split(",", maxsplit=1)
            var_location, var_criteria = (
                criteria_input_list[0],
                criteria_input_list[1].strip(),
            )
            variables = var_location.split("|")

            var_name = "config"
            dict_name = self.config
            while len(variables) > 1:
                var_name += f'["{variables[0]}"]'
                dict_name = dict_name.get(variables.pop(0))  # type: ignore

            var_name += f'["{variables[0]}"]'

            checker(var_name, dict_name[variables[0]], var_criteria)

    def _draw_standard_masks(  # pylint: disable-msg=too-many-locals
        self,
        image: np.ndarray,
        masks: np.ndarray,
        bbox_labels: np.ndarray,
    ) -> np.ndarray:
        """Draws instance segmentation masks over detected objects.

        Args:
            image (numpy.ndarray): Input image.
            masks (numpy.ndarray): NumPy array of binary (0/1) masks, one mask
                for each detected object, in the same order as bbox_labels.
            bbox_labels (numpy.ndarray): NumPy array of strings representing
                the labels of detected objects. The order corresponds to
                ``masks``.

        Returns:
            numpy.ndarray: Input image with instance segmentation masks
                applied to it.
        """
        self.class_instance_counts: Dict[str, int] = {}

        full_sized_canvas = np.zeros(image.shape, image.dtype)
        ret_image = image

        for index, _ in enumerate(bbox_labels):
            color = self._get_instance_color(bbox_labels[index])

            full_sized_canvas[:, :] = color

            coloured_seg_mask = cv2.bitwise_and(
                full_sized_canvas, full_sized_canvas, mask=masks[index]
            )
            masked_area = cv2.bitwise_and(image, image, mask=masks[index])
            masked_area_colored = cv2.addWeighted(
                coloured_seg_mask, ALPHA, masked_area, 1 - ALPHA, 0
            )

            # get the inverted mask i.e. image outside of the masked area
            mask_inv = 1 - masks[index]
            # remove masked area from image to be returned
            ret_image = cv2.bitwise_and(ret_image, ret_image, mask=mask_inv)
            ret_image = cv2.add(ret_image, masked_area_colored)

            if self.config["contours"]["show"]:
                ret_image = self._draw_contours(masks, ret_image, index)

        return ret_image

    def _get_instance_color(self, instance_class: str) -> Tuple[int, int, int]:
        """Returns color to use for next segmentation instance according to the
        chosen instance color scheme.

        Args:
            instance_class (str): Class of detected object.

        Returns:
            Tuple[int,int,int]: Color to use for next instance, in BGR.
        """
        self.class_instance_counts[instance_class] = (
            self.class_instance_counts.get(instance_class, 0) + 1
        )

        if self.class_instance_counts[instance_class] > len(
            self.class_instance_colors.setdefault(instance_class, [])
        ):
            # get new color
            color = self._get_new_instance_color(instance_class)
            # append new assigned color to instance_colors
            self.class_instance_colors[instance_class].append(color)
        else:
            # get color already assigned to this instance number
            color = self.class_instance_colors[instance_class][
                self.class_instance_counts[instance_class] - 1
            ]

        return color

    def _get_new_instance_color(self, instance_class: str) -> Tuple[int, int, int]:
        """Returns color to use for a new segmentation instance. When we
        encounter an instance that is one more than the count of instances of
        a particular class, the instance has not been assigned a color yet
        (from previous frames) and we need to get a new color.

        Depending on the chosen instance color scheme, this is how we determine
        the color:

        "random" - Random color.

        "hue_family" - Each object class uses a certain pre-assigned hue. Then
            each new instance will use the same hue, but with a slightly
            different saturation.

        Args:
            instance_class (str): Class of detected object.

        Returns:
            Tuple[int,int,int]: Color to use for next instance, in BGR.
        """
        if self.config["instance_color_scheme"] == "random":
            color_hsv = (randint(0, 179), randint(100, 255), 255)
            color = self._hsv_to_bgr(color_hsv)
        elif self.config["instance_color_scheme"] == "hue_family":
            if not self.class_instance_colors.get(instance_class):
                color = CLASS_COLORS.get(instance_class, DEFAULT_CLASS_COLOR)
            else:
                color = self.class_instance_colors.get(instance_class)[-1]  # type: ignore
                color_hsv = self._bgr_to_hsv(color)
                # we use a minimum saturation of 100 to avoid too light colors,
                # thus we increment saturation by step size of (256-100)/8.
                saturation = (
                    color_hsv[1] + (256 - SATURATION_MINIMUM) / SATURATION_STEPS
                ) % 256
                if saturation < SATURATION_MINIMUM:
                    saturation += SATURATION_MINIMUM
                color_hsv = (color_hsv[0], int(saturation), color_hsv[2])
                color = self._hsv_to_bgr(color_hsv)

        return color

    def _draw_contours(
        self,
        masks: np.ndarray,
        image: np.ndarray,
        index: int = None,
    ) -> np.ndarray:
        """Draws contours around instance segmentation masks. If 'index' is
        given, only the contour of the mask with the given index is drawn."""
        ret_image = image
        masks_to_process = [index] if index else range(masks.shape[0])
        for i in masks_to_process:
            contour, _ = cv2.findContours(
                masks[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                ret_image,
                contour,
                -1,
                CONTOUR_COLOR,
                self.config["contours"]["thickness"],
            )

        return ret_image

    def _mask_apply_effect(
        self, image: np.ndarray, masks: np.ndarray, effect: str
    ) -> np.ndarray:
        """Applies the chosen effect to the image and masks."""
        combined_masks = np.zeros(image.shape[:2], dtype="uint8")
        # combine all the individual masks
        for mask in masks:
            np.putmask(combined_masks, mask, 1)

        if self.config["effect_area"] == "objects":
            effect_area = combined_masks
        else:  # effect_area == "background"
            effect_area = 1 - combined_masks

        if effect in ["contrast", "brightness"]:
            full_image_with_effect = self._adjust_contrast_brightness(image, effect)
        elif effect == "gamma_correction":
            full_image_with_effect = self._gamma_correction(image)
        elif effect == "blur":
            full_image_with_effect = cv2.blur(
                image,
                (self.config["effect"]["blur"], self.config["effect"]["blur"]),
            )
        elif effect == "mosaic":
            full_image_with_effect = self._mosaic_image(image)

        effect_area_with_effect = cv2.bitwise_and(
            full_image_with_effect, full_image_with_effect, mask=effect_area
        )
        # get the inverted mask
        effect_area_inv = 1 - effect_area
        # remove effect area from original image
        image_empty_effect_area = cv2.bitwise_and(image, image, mask=effect_area_inv)
        # add both images together
        ret_image = cv2.add(image_empty_effect_area, effect_area_with_effect)

        if self.config["contours"]["show"]:
            ret_image = self._draw_contours(masks, ret_image)

        return ret_image

    def _adjust_contrast_brightness(self, image: np.ndarray, effect: str) -> np.ndarray:
        if effect == "contrast":
            alpha = self.config["effect"]["contrast"]
            beta = 0
        else:  # i.e. effect == "brightness"
            alpha = 1
            beta = self.config["effect"]["brightness"]

        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def _gamma_correction(self, image: np.ndarray) -> np.ndarray:
        lookup_table = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookup_table[0, i] = np.clip(
                pow(i / 255.0, self.config["effect"]["gamma_correction"]) * 255.0,
                0,
                255,
            )

        image_gamma_corrected = cv2.LUT(image, lookup_table)

        return image_gamma_corrected

    def _mosaic_image(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        ret_image = cv2.resize(
            image,
            (self.config["effect"]["mosaic"], self.config["effect"]["mosaic"]),
            interpolation=cv2.INTER_LANCZOS4,
        )
        ret_image = cv2.resize(
            ret_image, (width, height), interpolation=cv2.INTER_NEAREST
        )

        return ret_image

    @staticmethod
    def _bgr_to_hsv(bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
        hsv_norm = colorsys.rgb_to_hsv(bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)
        hsv = (int(hsv_norm[0] * 127), int(hsv_norm[1] * 255), int(hsv_norm[2] * 255))

        return hsv

    @staticmethod
    def _hsv_to_bgr(hsv: Tuple[int, int, int]) -> Tuple[int, int, int]:
        rgb_norm = colorsys.hsv_to_rgb(hsv[0] / 127, hsv[1] / 255, hsv[2] / 255)
        rgb = [int(x * 255) for x in rgb_norm]
        bgr = cast(Tuple[int, int, int], tuple(rgb[::-1]))

        return bgr
