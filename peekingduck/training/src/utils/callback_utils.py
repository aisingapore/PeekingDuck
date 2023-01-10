"""Callback Utils.

TODO:
1. Refactor `init_improvement` as it is referenced from torchflare.
"""
import math
from functools import partial
import torch
from typing import Callable, Tuple


def _is_min(
    curr_epoch_score: torch.Tensor, curr_best_score: torch.Tensor, min_delta: float
) -> bool:
    return curr_epoch_score <= (curr_best_score - min_delta)


def _is_max(
    curr_epoch_score: torch.Tensor, curr_best_score: torch.Tensor, min_delta: float
) -> bool:
    return curr_epoch_score >= (curr_best_score + min_delta)


def init_improvement(mode: str, min_delta: float) -> Tuple[Callable, torch.Tensor]:
    """Get the scoring function and the best value according to mode.

    Args:
        mode (str): One of {"min", "max"}.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.

    Returns:
        improvement (Callable): Function to check if the score is an improvement.
        best_score (torch.Tensor): Initialize the best score as either -inf or inf depending on mode.
    """
    if mode == "min":
        improvement = partial(_is_min, min_delta=min_delta)
        best_score = math.inf
    else:
        improvement = partial(_is_max, min_delta=min_delta)
        best_score = -math.inf
    return improvement, best_score
