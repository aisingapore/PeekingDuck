# Copyright 2021 AI Singapore
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
#
# Code of this file is mostly forked from
# [@xuannianz](https://github.com/xuannianz))


"""
Anchor-related functions for EfficientDet
"""


from typing import Tuple, List, Callable
import numpy as np
from tensorflow import keras


class AnchorParameters:
    """
    The parameters that define how anchors are generated.

    Args
        sizes : List of sizes to use. Each size corresponds to one feature level.
        strides : List of strides to use. Each stride correspond to one feature level.
        ratios : List of ratios to use per location in a feature map.
        scales : List of scales to use per location in a feature map.
    """

    def __init__(self, sizes: Tuple[int, int, int, int, int] = (32, 64, 128, 256, 512),
                 strides: Tuple[int, int, int, int, int] = (8, 16, 32, 64, 128),
                 ratios: Tuple[float, float, float] = (1.0, 0.5, 2.0),
                 scales: Tuple[float, float, float] = (1.0, 2**(1. / 3.), 2**(2. / 3.))) -> None:
        self.sizes = sizes
        self.strides = strides
        self.ratios = np.array(ratios, dtype=keras.backend.floatx())
        self.scales = np.array(scales, dtype=keras.backend.floatx())

    def num_anchors(self) -> int:
        """Function to get number of anchors

        Returns:
            int: the total number of anchors
        """
        return len(self.ratios) * len(self.scales)

    def get_sizes(self) -> Tuple[int, int, int, int, int]:
        """Getter for sizes
        """
        return self.sizes


# The default anchor parameters.
AnchorParameters.default = AnchorParameters(  # type: ignore
    sizes=(32, 64, 128, 256, 512),
    strides=(8, 16, 32, 64, 128),
    ratios=np.array([1, 0.5, 2], keras.backend.floatx()),
    scales=np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
)


def guess_shapes(image_shape: Tuple[int, int], pyramid_levels: List[int]) -> List[List]:
    """
    Guess shapes based on pyramid levels.

    Args
         image_shape: The shape of the image.
         pyramid_levels: A list of what pyramid levels are used.

    Returns
        A list of image shapes at each pyramid level.
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def generate_anchors(base_size: int = 16,
                     ratios: Tuple[float] = None,
                     scales: Tuple[float] = None) -> np.ndarray:
    """
    Generate anchor windows by enumerating aspect ratios X scales w.r.t. a reference window.

    Args:
        base_size:
        ratios:
        scales:

    Returns:

    """
    if ratios is None:
        ratios = AnchorParameters.default.ratios  # type: ignore

    if scales is None:
        scales = AnchorParameters.default.scales  # type: ignore

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    anchors[:, 2:] = base_size * np.tile(np.repeat(scales, len(ratios))[None], (2, 1)).T

    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.tile(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.tile(ratios, len(scales))

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def shift(feature_map_shape: List[float], stride: int, anchors: np.ndarray) -> np.ndarray:
    """
    Produce shifted anchors based on shape of the map and stride size.

    Args
        feature_map_shape : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """

    # create a grid starting from half stride from the top left corner
    shift_x = (np.arange(0, feature_map_shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, feature_map_shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    a_num = anchors.shape[0]
    k_num = shifts.shape[0]
    all_anchors = (anchors.reshape((1, a_num, 4)) +
                   shifts.reshape((1, k_num, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((k_num * a_num, 4))

    return all_anchors


def anchors_for_shape(
        image_shape: Tuple[int, int],
        pyramid_levels: List[int] = None,
        anchor_params: AnchorParameters = None,
        shapes_callback: Callable = None,
) -> np.ndarray:
    """
    Generators anchors for a given shape.

    Args
        image_shape: The shape of the image.
        pyramid_levels: List representing which pyramids to use (defaults to [3, 4, 5, 6, 7]).
        anchor_params: Struct containing anchor parameters. If None, default values are used.
        shapes_callback: Function to call to get the shape of image at different pyramid levels.

    Returns
        np.array of shape (N, 4) containing the (x1, y1, x2, y2) coordinates for the anchors.
    """

    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]

    if anchor_params is None:
        anchor_params = AnchorParameters.default  # type: ignore

    if shapes_callback is None:
        shapes_callback = guess_shapes
    feature_map_shapes = shapes_callback(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4), dtype=np.float32)
    for idx, _ in enumerate(pyramid_levels):
        anchors = generate_anchors(
            base_size=anchor_params.sizes[idx],
            ratios=anchor_params.ratios,
            scales=anchor_params.scales
        )
        shifted_anchors = shift(feature_map_shapes[idx], anchor_params.strides[idx], anchors)
        all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors.astype(np.float32)
