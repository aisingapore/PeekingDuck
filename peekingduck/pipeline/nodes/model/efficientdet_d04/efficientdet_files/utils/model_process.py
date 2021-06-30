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
Processing helper functinos for EfficientDet
"""

from typing import List, Tuple
import numpy as np
import cv2

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


def preprocess_image(image: np.ndarray,
                     image_size: int) -> Tuple[List[List[float]], float]:
    """Preprocessing helper function for efficientdet

    Args:
        image (np.array): the input image in numpy array
        image_size (int): the model input size as specified in config

    Returns:
        image (np.array): the preprocessed image
        scale (float): the scale in which the original image was resized to
    """
    # image, RGB
    image_height, image_width = image.shape[:2]
    if image_height > image_width:
        scale = image_size / image_height
        resized_height = image_size
        resized_width = int(image_width * scale)
    else:
        scale = image_size / image_width
        resized_height = int(image_height * scale)
        resized_width = image_size

    image = cv2.resize(image, (resized_width, resized_height))
    image = image.astype(np.float32)
    image /= 255.
    image -= IMG_MEAN
    image /= IMG_STD
    pad_h = image_size - resized_height
    pad_w = image_size - resized_width
    image = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant')

    return image, scale


def postprocess_boxes(boxes: np.ndarray,
                      scale: float,
                      height: int,
                      width: int) -> np.ndarray:
    """Postprocessing helper function for efficientdet

    Args:
        boxes (np.array): the original detected bboxes from model output
        scale (float): scale in which the original image was resized to
        height (int): the height of the original image
        width (int): the width of the original image

    Returns:
        boxes (np.array): the postprocessed bboxes
    """
    boxes /= scale
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)

    boxes[:, [0, 2]] /= width
    boxes[:, [1, 3]] /= height
    return boxes
