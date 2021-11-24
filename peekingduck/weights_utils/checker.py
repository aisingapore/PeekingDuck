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
Checks for model weights in weights folder.
"""

from pathlib import Path


def has_weights(weights_dir: Path, model_dir: Path) -> bool:
    """Checks for model weight paths from weights folder.

    Args:
        weights_dir (:obj:`Path`): Path to where all weights are stored.
        model_dir (:obj:`Path`): Path to where weights for a model are stored.

    Returns:
        (:obj:`bool`): ``True`` if specified files/directories in
        ``weights_dir`` exist, else ``False``.
    """
    if not weights_dir.exists():
        weights_dir.mkdir()
        return False
    return model_dir.exists()
