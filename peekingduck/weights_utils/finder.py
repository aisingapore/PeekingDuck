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
Finds paths for where weights are stored.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

PEEKINGDUCK_WEIGHTS_DIR = "peekingduck_weights"


def find_paths(
    root: Path, weights_parent_dir: str, weights: Dict[str, Any]
) -> Tuple[Path, Path]:
    """Checks for model weight paths from weights folder.

    Args:
        root (:obj:`str`): Path of ``peekingduck`` root folder.
        weights_parent_dir (:obj:`str`): Parent dir of where weights will be stored.
        weights (:obj:`Dict[str, Any]`): File/subdir names of weights.

    Returns:
        weights_dir (:obj:`pathlib.Path`): Path to where all weights are stored.
        model_dir (:obj:`pathlib.Path`): Path to where weights for a model are stored.
    """

    if weights_parent_dir.lower() == "default":
        weights_dir = root.parent / PEEKINGDUCK_WEIGHTS_DIR
    else:
        if not Path(weights_parent_dir).exists():
            raise FileNotFoundError(
                f"The specified weights_parent_dir: {weights_parent_dir} does not exist."
            )
        weights_dir = Path(weights_parent_dir) / PEEKINGDUCK_WEIGHTS_DIR

    model_dir = weights_dir / weights["model_subdir"]

    return weights_dir, model_dir
