"""
Copyright 2021 AI Singapore

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import yaml
from pathlib import Path
from unittest import mock

import pytest

from peekingduck.pipeline.nodes.model.hrnetv1.hrnet_model import HRNetModel


def hrnet_config():
    filepath = (
        Path.cwd()
        / "tests"
        / "pipeline"
        / "nodes"
        / "model"
        / "hrnetv1"
        / "test_hrnet.yml"
    )
    with open(filepath) as file:
        node_config = yaml.safe_load(file)
    node_config["root"] = Path.cwd()

    return node_config


@pytest.mark.mlmodel
class TestHrnetModel:
    @mock.patch("peekingduck.weights_utils.checker.has_weights")
    @mock.patch("builtins.print")
    def test_no_weight(self, mock_print, mock_has_weights):
        mock_has_weights.return_value = False

        msg_1 = "---no hrnet weights detected. proceeding to download...---"
        msg_2 = "---hrnet weights download complete.---"

        config = hrnet_config()
        HRNetModel(config)

        assert mock_print.mock_calls == [mock.call(msg_1), mock.call(msg_2)]
