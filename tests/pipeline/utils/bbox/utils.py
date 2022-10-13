from peekingduck.pipeline.utils.bbox.transforms import BboxType, clone


def expand_dim(
    bboxes: BboxType,
    num_dims: int,
) -> BboxType:
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
