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
from unittest import mock

import pytest

from peekingduck.pipeline.nodes.base import (
    PEEKINGDUCK_WEIGHTS_SUBDIR,
    WeightsDownloaderMixin,
)
from tests.conftest import PKD_DIR


@pytest.fixture
def weights_model():
    return WeightsModel()


class WeightsModel(WeightsDownloaderMixin):
    def __init__(self):
        self.config = {}
        self.logger = mock.Mock()


class TestBase:
    def test_parent_dir_not_exist(self, weights_model):
        invalid_dir = "invalid_dir"
        weights_model.config["weights_parent_dir"] = invalid_dir

        with pytest.raises(FileNotFoundError) as excinfo:
            weights_model.download_weights()
        assert (
            f"The specified weights_parent_dir: {invalid_dir} does not exist."
            == str(excinfo.value)
        )

    def test_parent_dir_not_absolute(self, weights_model):
        relative_dir = PKD_DIR.relative_to(PKD_DIR.parent)
        weights_model.config["weights_parent_dir"] = relative_dir

        with pytest.raises(ValueError) as excinfo:
            weights_model.download_weights()
        assert (
            f"The specified weights_parent_dir: {relative_dir} must be an absolute path."
            == str(excinfo.value)
        )

    @pytest.mark.usefixtures("tmp_dir")
    def test_create_weights_dir(self, weights_model, replace_download_weights):
        weights_parent_dir = Path.cwd().resolve()
        weights_model.config["weights_parent_dir"] = weights_parent_dir
        weights_model.config["weights"] = {"model_subdir": "some_model"}

        assert not (weights_parent_dir / PEEKINGDUCK_WEIGHTS_SUBDIR).exists()

        with mock.patch.object(
            WeightsDownloaderMixin, "_download_blob_to", wraps=replace_download_weights
        ), mock.patch.object(
            WeightsDownloaderMixin, "extract_file", wraps=replace_download_weights
        ):
            weights_model.download_weights()

        assert (weights_parent_dir / PEEKINGDUCK_WEIGHTS_SUBDIR).exists()
