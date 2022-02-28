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

from pathlib import Path

import pytest

from peekingduck.weights_utils.finder import find_paths

FAKE_ROOT_PATH = Path("/level_one/level_two/level_three")
FAKE_ROOT_PARENT_PATH = Path("/level_one/level_two")
CURRENT_PATH = Path.cwd()


@pytest.fixture
def weights():
    return {
        "model_subdir": "new_model",
        "blob_file": "new_model.zip",
        "classes_file": "new_model.json",
    }


class TestFinder:
    def test_default_dir(self, weights):
        weights_dir, model_dir = find_paths(FAKE_ROOT_PATH, weights, None)
        assert weights_dir == Path(FAKE_ROOT_PARENT_PATH / "peekingduck_weights")
        assert model_dir == Path(
            FAKE_ROOT_PARENT_PATH / "peekingduck_weights/new_model"
        )

    def test_parent_path_not_exist(self, weights):
        with pytest.raises(FileNotFoundError):
            find_paths(FAKE_ROOT_PATH, weights, "no_exist_path")

    def test_parent_path_not_absolute(self, weights):
        with pytest.raises(ValueError):
            find_paths(
                FAKE_ROOT_PATH,
                weights,
                ".",
            )
