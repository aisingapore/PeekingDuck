# Copyright 2023 AI Singapore
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

"""Callback Utils.
1. Refactor `init_improvement` as it is referenced from torchflare.
"""

from typing import Any, Callable, Tuple
from functools import partial
import math
import torch


def _is_min(
    curr_epoch_score: torch.Tensor, curr_best_score: torch.Tensor, min_delta: float
) -> bool:
    return curr_epoch_score <= (curr_best_score - min_delta)  # type: ignore


def _is_max(
    curr_epoch_score: torch.Tensor, curr_best_score: torch.Tensor, min_delta: float
) -> bool:
    return curr_epoch_score >= (curr_best_score + min_delta)  # type: ignore


def init_improvement(mode: str, min_delta: float) -> Tuple[Callable[..., Any], float]:
    """Get the scoring function and the best value according to mode.

    Args:
        mode (str): One of {"min", "max"}.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.

    Returns:
        improvement (Callable): Function to check if the score is an improvement.
        best_score (float): Initialize the best score as either -inf or inf depending on mode.
    """
    if mode == "min":
        improvement = partial(_is_min, min_delta=min_delta)
        best_score = math.inf
    else:
        improvement = partial(_is_max, min_delta=min_delta)
        best_score = -math.inf
    return improvement, best_score
