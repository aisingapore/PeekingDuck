import numpy as np
import torch
from typing import Union
from peekingduck.pipeline.utils.bbox.transforms import clone


def list2numpy(list_: list) -> np.ndarray:
    """Convert list to numpy array."""
    return np.asarray(list_)


def list2torch(list_: list) -> torch.Tensor:
    """Convert list to torch tensor."""
    return torch.tensor(list_)


def expand_dim(
    bboxes: Union[np.ndarray, torch.Tensor],
    num_dims: int,
) -> Union[np.ndarray, torch.Tensor]:
    """Expand the dimension of bboxes (first in) by num_dims.

    Note:
        np.expand_dims will not work for tuple dim numpy < 1.18.0 which
        is not the version in our cicd.

    Args:
        bboxes (Union[np.ndarray, torch.Tensor]): The input bboxes.
        num_dims (int): The number of dimensions to expand.

    Returns:
        (Union[np.ndarray, torch.Tensor]): The bboxes with expanded dimensions.
    """
    bboxes = clone(bboxes)
    return bboxes[(None,) * num_dims]
