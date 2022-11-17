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
Preprocessing functions for HRNet
"""

from typing import Tuple

import cv2
import numpy as np


def tlwh2xywh(inputs: np.ndarray, aspect_ratio: float) -> np.ndarray:
    """Converts from [t, l, w, h] to [x, y, w1, h] format.

    (t, l) is the coordinates of the top left corner, w is the width, and h is
    the height. (x, y) is the coordinates of the center. The bounding box is
    adjusted to meet the input aspect ratio, w1 is the new width and h1 is the
    new height.

    [x, y, w1, h1] is calculated as:
    x = t + w * 0.5
    y = l + h * 0.5
    w1 = {
        h * aspect_ratio, if w < h * aspect_ratio
        w               , otherwise
    }
    h1 = {
        w / aspect_ratio, if w > h * aspect_ratio
        h               , otherwise
    }

    Args:
        inputs (np.ndarray): Input bounding boxes (2-d array) each with the
            format `(top left x, top left y, width, height)`.
        bboxes (np.ndarray): Array of bboxes in the form of top left
            (x, y, w, h).
        aspect_ratio (float): w:h ratio for the new box.

    Returns:
        (np.ndarray): Array of bboxes in center (x, y, w1, h1) format that
        meets the aspect ratio.
    """
    outputs = np.empty_like(inputs)
    outputs[:, 0] = inputs[:, 0] + inputs[:, 2] * 0.5
    outputs[:, 1] = inputs[:, 1] + inputs[:, 3] * 0.5
    outputs[:, 2] = inputs[:, 2]
    outputs[:, 3] = inputs[:, 3]

    target_width = inputs[:, 3] * aspect_ratio

    tall_bboxes = inputs[:, 2] < target_width
    outputs[tall_bboxes, 2] = inputs[tall_bboxes, 3] * aspect_ratio

    wide_bboxes = inputs[:, 2] > target_width
    outputs[wide_bboxes, 3] = inputs[wide_bboxes, 2] / aspect_ratio

    return outputs


def crop_and_resize(
    frame: np.ndarray, bboxes: np.ndarray, out_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop a region from frame specified by its center and size. The
    cropped region is resized to out_size.

    Args:
        frame (np.ndarray): Image in numpy array.
        bboxes (np.ndarray): Bboxes center (x, y, w, h) coordinates.
        out_size (tuple): Cropped region will be resized to out_size.

    Returns:
        (Tuple[np.ndarray, np.ndarray]): The resized and cropped region array
        and the affine transform matrix to map a point in cropped image
        coordinate space to source frame coordinate space.
    """
    translate_x = bboxes[:, 0] - (bboxes[:, 2] - 1) * 0.5
    translate_y = bboxes[:, 1] - (bboxes[:, 3] - 1) * 0.5
    scale_x = bboxes[:, 2] / out_size[0]
    scale_y = bboxes[:, 3] / out_size[1]
    zero_mat = np.zeros((len(translate_x),))

    x_mat = np.column_stack((scale_x, zero_mat, translate_x))
    y_mat = np.column_stack((zero_mat, scale_y, translate_y))
    affine_matrices = np.concatenate((x_mat, y_mat), axis=1)
    affine_matrices = affine_matrices.reshape((-1, 2, 3))

    transformed_images = [
        cv2.warpAffine(
            frame, x, out_size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        )
        for x in affine_matrices
    ]
    return transformed_images, affine_matrices
