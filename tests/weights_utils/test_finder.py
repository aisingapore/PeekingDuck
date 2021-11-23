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
