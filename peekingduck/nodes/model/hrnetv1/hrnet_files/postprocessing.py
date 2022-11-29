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
Postprocessing functions for HRNet
"""

import numpy as np


def affine_transform_xy(
    keypoints: np.ndarray, affine_matrices: np.ndarray
) -> np.ndarray:
    """Apply respective affine transform on array of points.

    Args:
        keypoints (np.array): array sequence of points (x, y)
        affine_matrices (np.array) - array of 2x3 affine transform matrix

    Returns:
        array of transformed points
    """
    transformed_matrices = []
    keypoints = np.dstack(
        (keypoints, np.ones((keypoints.shape[0], keypoints.shape[1], 1)))
    )
    for affine_matrix, keypoint in zip(affine_matrices, keypoints):
        transformed_keypoint = np.dot(affine_matrix, keypoint.T)
        transformed_matrices.append(transformed_keypoint.T)

    return np.array(transformed_matrices)
