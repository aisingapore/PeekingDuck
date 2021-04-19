import os


def has_weights(root, path_to_check):
    """Checks for model weight paths from weights folder

    Args:
        root (str): path of peekingduck root folder
        path_to_check (List[str]): list of files/directories to check
            to see if weights exists

    Returns:
        boolean: True is files/directories needed exists, else False
    """

    # Check for whether weights dir even exist. If not make directory
    # Empty directory should then return False
    weights_dir = os.path.join(root, '..', 'weights')
    if not os.path.isdir(weights_dir):
        os.mkdir(weights_dir)
        return False

    for check in path_to_check:
        if not os.path.exists(os.path.join(root, check)):
            return False
    return True