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
#
# Code of this file is mostly forked from
# [@xuannianz](https://github.com/xuannianz))

"""
EfficientDet auxiliary functions (compiled with numba jit)
"""

import numpy as np
from numba import jit_module

from peekingduck.nodes.model.efficientdet_d04.efficientdet_files.constants import (
    IMG_MEAN,
    IMG_STD,
)


def normalize_and_pad(image: np.ndarray, pad_height: int, pad_width: int) -> np.ndarray:
    """Normalizes image values and pad the right and bottom of the image.

    Args:
        image (np.ndarray): The input image with uint8 pixel values.
        pad_height (int): The amount of vertical padding.
        pad_width (int): The amount of horizontal padding.

    Returns:
        (np.ndarray): The padded image with normalized values.
    """
    image = image.astype(np.float32)
    image = (image / 255.0 - IMG_MEAN) / IMG_STD
    padded_image = np.zeros(
        (image.shape[0] + pad_height, image.shape[1] + pad_width, image.shape[2]),
        dtype=np.float32,
    )
    padded_image[: image.shape[0], : image.shape[1]] = image
    return np.expand_dims(padded_image, axis=0)


jit_module(nopython=True)
