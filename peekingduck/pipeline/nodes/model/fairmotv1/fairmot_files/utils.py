# Modifications copyright 2022 AI Singapore
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
# Original copyright (c) 2020 YifuZhang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Utility functions used by FairMOT.

Modification:
- Change _get_affine_transform to always use the same x- and y- scaling
- Change _get_affine_transform to accept np.ndarray for output_size
- Use @ instead of np.dot for matrix multiplication
"""

from typing import Tuple

import cv2
import numpy as np
import torch


def gather_feat(feat: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gathers values specified by `indices` along dim=1.

    Args:
        feat (torch.Tensor): The source tensor containing the features of
            interest.
        indices (torch.Tensor): The indices of elements to gather.

    Returns:
        (torch.Tensor): The gathered features.
    """
    dim = feat.size(2)
    indices = indices.unsqueeze(2).expand(indices.size(0), indices.size(1), dim)
    feat = feat.gather(1, indices)
    return feat


def letterbox(
    image: np.ndarray,
    height: int,
    width: int,
    colour: Tuple[float, float, float] = (127.5, 127.5, 127.5),
) -> np.ndarray:
    """Resizes a rectangular image to a padded rectangular image.

    Args:
        image (np.ndarray): Image frame.
        height (int): Height of padded image.
        width (int): Width of padded image.
        colour (Tuple[float, float, float]): Colour used for padding around
            the image. (127.5, 127.5, 127.5) is chosen as it is used by the
            original project during model training.

    Returns:
        (np.ndarray): Padded rectangular image.
    """
    shape = image.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    # new_shape = [width, height]
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    width_padding = (width - new_shape[0]) / 2
    height_padding = (height - new_shape[1]) / 2
    top = round(height_padding - 0.1)
    bottom = round(height_padding + 0.1)
    left = round(width_padding - 0.1)
    right = round(width_padding + 0.1)
    # resized, no border
    image = cv2.resize(image, new_shape, interpolation=cv2.INTER_AREA)
    # padded rectangular
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=colour
    )
    return image


def transform_coords(
    coords: np.ndarray,
    center: np.ndarray,
    scale: float,
    output_size: Tuple[float, float],
) -> np.ndarray:
    """Translates and scales the coordinate w.r.t. the original image size.

    Args:
        coords (np.ndarray): Coordinates of a bbox corner.
        center (np.ndarray): Coordinates of the center of the original image.
        scale (float): Scale between the input image fed to the model and the
            original image.
        output_size(Tuple[float, float]): Output image size by the model.

    Returns:
        (np.ndarray): Transformed coordinates.
    """
    target_coords = np.zeros_like(coords)
    matrix = _get_affine_transform(
        center, scale, 0, np.array(output_size, dtype=np.float32), inv=True
    )
    for i in range(coords.shape[0]):
        target_coords[i, :2] = _affine_transform(coords[i, :2], matrix)
    return target_coords


def transpose_and_gather_feat(
    feat: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Transposes the features and then gather the features specified by
    `indices` at dim=1.

    Args:
        feat (torch.Tensor): The tensor containing the features of interest.
        indices (torch.Tensor): The indices of elements to gather.

    Returns:
        (torch.Tensor): The transposed and gathered features.
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, indices)
    return feat


def _affine_transform(point: np.ndarray, trans_matrix: np.ndarray) -> np.ndarray:
    """Applies affine transformation to the specified coordinates.

    Args:
        point (np.ndarray): The coordinates to transform.
        trans_matrix (np.ndarray): A 2x3 affine transformation matrix.

    Returns:
        (np.ndarray): The transformed coordinates.
    """
    new_pt = np.array([point[0], point[1], 1.0], dtype=np.float32).T
    new_pt = trans_matrix @ new_pt
    return new_pt[:2]


def _get_3rd_point(point_1: np.ndarray, point_2: np.ndarray) -> np.ndarray:
    """Computes the third point required for computing affine transformation
    matrix.

    Args:
        point_1 (np.ndarray): The first point.
        point_2 (np.ndarray): The second point.

    Returns:
        (np.ndarray): The third point.
    """
    direct = point_1 - point_2
    return point_2 + np.array([-direct[1], direct[0]], dtype=np.float32)


def _get_affine_transform(  # pylint:disable=too-many-arguments
    center: np.ndarray,
    scale: float,
    rot: float,
    output_size: np.ndarray,
    shift: np.ndarray = np.array([0, 0], dtype=np.float32),
    inv: bool = False,
) -> np.ndarray:
    """Computes the affine transformation matrix described by the given
    arguments.

    Args:
        center (np.ndarray): Coordinates of the center.
        scale (float): Scale factor.
        rot (float): Rotation amount (in degrees).
        output_size (np.ndarray): Height and width of the output.
        shift (np.ndarray): Translation.
        inv (bool): Flag to determine if we should perform the inverse of the
            transformation.

    Returns:
        (np.ndarray): The affine transformation matrix.
    """
    src_w = scale
    dst_w = output_size[0]

    src_dir = _get_dir([0, src_w * -0.5], np.pi * rot / 180)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = output_size * 0.5
    dst[1, :] = output_size * 0.5 + dst_dir

    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        matrix = cv2.getAffineTransform(dst, src)
    else:  # pragma: no cover
        # Only inv=True is used by FairMOT
        matrix = cv2.getAffineTransform(src, dst)

    return matrix


def _get_dir(src_point: np.ndarray, rot_rad: float) -> np.ndarray:
    """Computes the direction vector.

    Args:
        src_point (np.ndarray): The coordinates of the source point.
        rot_rad (float): The amount of rotation (in radians).

    Returns:
        (np.ndarray): The direction vector.
    """
    sin = np.sin(rot_rad)
    cos = np.cos(rot_rad)

    src_result = np.array(
        [
            src_point[0] * cos - src_point[1] * sin,
            src_point[0] * sin + src_point[1] * cos,
        ]
    )

    return src_result
