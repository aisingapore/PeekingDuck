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

import os
import yaml
from unittest import mock
from unittest.mock import call

import pytest
from peekingduck.pipeline.nodes.model.hrnetv1.hrnet_model import HRNetModel


def hrnet_config():
    filepath = os.path.join(os.getcwd(), 'tests/pipeline/nodes/model/hrnetv1/test_hrnet.yml')
    with open(filepath) as file:
        node_config = yaml.safe_load(file)
    node_config['root'] = os.getcwd()
    return node_config


@pytest.mark.mlmodel
class TestHrnetModel:

    def test_no_weight(self):

        with mock.patch('peekingduck.weights_utils.checker.has_weights',
                        return_value=False):
            with mock.patch('builtins.print') as mocked_print:

                msg_1 = '---no hrnet weights detected. proceeding to download...---'
                msg_2 = '---hrnet weights download complete.---'

                config = hrnet_config()
                HRNetModel(config)

                assert mocked_print.mock_calls == [call(msg_1), call(msg_2)]
