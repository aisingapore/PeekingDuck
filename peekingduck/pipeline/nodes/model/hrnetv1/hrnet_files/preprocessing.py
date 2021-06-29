# Copyright 2021 AI Singapore
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
import numpy as np
import cv2


def project_bbox(bboxes: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Project the normalized bbox to specified image coordinates.

    Args:
        bboxes (np.array): Array of detected bboxes in (top left, btm right) format
        size (tuple): width and height of image space

    Returns:
        array of bboxes in (x, y, w, h) format where x,y is top left coordinate
    """
    width, height = size[0] - 1, size[1] - 1

    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, 1)
    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, 1)
    bboxes[:, 2] = np.clip(bboxes[:, 2], bboxes[:, 0], 1)
    bboxes[:, 3] = np.clip(bboxes[:, 3], bboxes[:, 1], 1)

    bboxes[:, [0, 2]] *= width
    bboxes[:, [1, 3]] *= height

    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]

    return bboxes


def box2cs(bboxes: np.ndarray, aspect_ratio: float) -> np.ndarray:
    """Convert bounding box defined by top left x, y, w, h to its center x,y,w,h
    The bounding box is also expanded to meet the input aspect ratio.

    Args:
        bboxes (np.array): Array of bboxes in of x, y, w, h (top left)
        aspect_ratio(float): W:H ratio for the new box

    Returns:
        Array of bboxes in x, y, w, h (center) format that meets the aspect ratio
    """

    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5

    req_w = aspect_ratio * bboxes[:, 3]
    bboxes[:, 3][bboxes[:, 2] > req_w] = bboxes[bboxes[:, 2] > req_w][:, 2] / aspect_ratio
    bboxes[:, 2][bboxes[:, 2] < req_w] = bboxes[bboxes[:, 2] < req_w][:, 3] * aspect_ratio

    return bboxes


def crop_and_resize(frame: np.ndarray,
                    bboxes: np.ndarray,
                    out_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Crop a region from frame specified by its center and size. The
    cropped region is resized to out_size.

    Args:
        frame (np.array): Image in numpy array
        bboxes (np.array): bboxes (x,y,w,h) center coordinates
        out_size (tuple): cropped region will be resized to out_size

    return:
        the resized and cropped region array and the affine transform matrix
        to map a point in cropped image coordinate space to source frame
        coordinate space
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

    transformed_images = [cv2.warpAffine(frame,
                                         x,
                                         out_size,
                                         flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                          for x in affine_matrices]
    return transformed_images, affine_matrices
