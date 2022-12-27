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
Processing helper functions for EfficientDet
"""

from typing import Callable, Tuple

import cv2
import numpy as np


def preprocess_image(
    image: np.ndarray, image_size: int, normalize_and_pad: Callable
) -> Tuple[np.ndarray, float]:
    """Preprocessing helper function for efficientdet

    Args:
        image (np.ndarray): the input image in numpy array
        image_size (int): the model input size as specified in config

    Returns:
        image (np.ndarray): the preprocessed image
        scale (float): the scale in which the original image was resized to
    """
    # image, RGB
    height, width = image.shape[:2]
    scale = image_size / max(height, width)
    if height > width:
        resized_height = image_size
        resized_width = int(width * scale)
    else:
        resized_height = int(height * scale)
        resized_width = image_size
    pad_height = image_size - resized_height
    pad_width = image_size - resized_width

    image = cv2.resize(image, (resized_width, resized_height))
    image = normalize_and_pad(image, pad_height, pad_width)

    return image, scale


def postprocess_boxes(
    boxes: np.ndarray, scale: float, height: int, width: int
) -> np.ndarray:
    """Postprocessing helper function for efficientdet

    Args:
        boxes (np.ndarray): the original detected bboxes from model output
        scale (float): scale in which the original image was resized to
        height (int): the height of the original image
        width (int): the width of the original image

    Returns:
        boxes (np.ndarray): the postprocessed bboxes
    """
    boxes /= scale
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)

    boxes[:, [0, 2]] /= width
    boxes[:, [1, 3]] /= height
    return boxes
