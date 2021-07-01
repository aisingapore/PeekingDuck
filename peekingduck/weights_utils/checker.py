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
Checks for model weights in weights folder
"""


import os
from typing import List


def has_weights(root: str, path_to_check: List[str]) -> bool:
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
