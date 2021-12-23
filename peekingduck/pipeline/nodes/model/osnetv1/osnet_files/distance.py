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
#
# Original copyright (c) 2018 Kaiyang Zhou
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

"""
Compute distance between query and gallery features.
"""

from __future__ import division, print_function, absolute_import
import torch
from torch.nn import functional as F


def compute_distance_matrix(
    input1: torch.Tensor, input2: torch.Tensor, metric: str = "euclidean"
) -> torch.Tensor:
    """A wrapper function for computing distance matrix.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        metric (str): "euclidean" or "cosine".
            Default is "euclidean".

    Raises:
        ValueError: Unknown distance metric chosen.

    Returns:
        torch.Tensor: Distance matrix.
    """
    # Check input
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)
    assert input1.dim() == 2, f"Expected 2-D tensor, but got {input1.dim()}-D"
    assert input2.dim() == 2, f"Expected 2-D tensor, but got {input2.dim()}-D"
    assert input1.size(1) == input2.size(1)

    if metric == "euclidean":
        distmat = euclidean_squared_distance(input1, input2)
    elif metric == "cosine":
        distmat = cosine_distance(input1, input2)
    else:
        raise ValueError(
            f"Unknown distance metric: {metric}. "
            "Please choose either 'euclidean' or 'cosine'"
        )

    return distmat


def euclidean_squared_distance(
    input1: torch.Tensor, input2: torch.Tensor
) -> torch.Tensor:
    """Computes euclidean squared distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: Distance matrix.
    """
    input1_size = input1.size(0)
    input2_size = input2.size(0)
    mat1 = (
        torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(input1_size, input2_size)
    )
    mat2 = (
        torch.pow(input2, 2)
        .sum(dim=1, keepdim=True)
        .expand(input2_size, input1_size)
        .t()
    )
    distmat = mat1 + mat2
    distmat.addmm_(input1, input2.t(), beta=1, alpha=-2)
    return distmat


def cosine_distance(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """Computes cosine distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: Distance matrix.
    """
    input1_norm = F.normalize(input1, p=2, dim=1)
    input2_norm = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_norm, input2_norm.t())
    return distmat
