# Copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pathlib import Path

import pytest
import yaml

from peekingduck.pipeline.nodes.model.hrnetv1.hrnet_files.detector import Detector
from peekingduck.weights_utils import checker, downloader


@pytest.fixture
def hrnet_config():
    with open(Path(__file__).resolve().parent / "test_hrnet.yml") as file:
        node_config = yaml.safe_load(file)
    node_config["root"] = Path.cwd()

    return node_config


@pytest.fixture
def model_dir(hrnet_config):
    return (
        hrnet_config["root"]
        / "peekingduck_weights"
        / hrnet_config["weights"]["model_subdir"]
    )


@pytest.fixture
def weights_dir(hrnet_config):
    return hrnet_config["root"] / "peekingduck_weights"


@pytest.mark.mlmodel
class TestDetector:
    def test_create_model(self, hrnet_config, model_dir, weights_dir):
        """Testing hrnet model instantiation."""
        if not checker.has_weights(weights_dir, model_dir):
            downloader.download_weights(
                weights_dir, hrnet_config["weights"]["blob_file"]
            )

        hrnet_detector = Detector(hrnet_config, model_dir)
        hrnet_model = hrnet_detector._create_hrnet_model
        assert hrnet_model is not None
