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
import pytest
from peekingduck.pipeline.nodes.model.hrnetv1.hrnet_files.detector import Detector


@pytest.fixture
def hrnet_config():
    filepath = os.path.join(os.getcwd(), 'tests/pipeline/nodes/model/hrnetv1/test_hrnet.yml')
    with open(filepath) as file:
        node_config = yaml.safe_load(file)
    node_config['root'] = os.getcwd()
    return node_config


@pytest.fixture
def hrnet_detector(hrnet_config):
    detector = Detector(hrnet_config)
    return detector


@pytest.mark.mlmodel
class TestDetector:

    def test_create_model(self, hrnet_detector):
        """Testing hrnet model instantiation.
        """
        hrnet_model = hrnet_detector._create_hrnet_model
        assert hrnet_model is not None
