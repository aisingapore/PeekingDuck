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
from peekingduck.pipeline.nodes.model.posenet import Node
import pytest
import yaml
import cv2
import os
import numpy as np
import numpy.testing as npt


TEST_DIR = os.path.join(os.getcwd(), 'images', 'testing')
models = [50, 75, 100, 'resnet']
person_image_list = ['t1.jpg', 't2.jpg', 't4.jpg']
empty_image_list = ['black.jpg', 't3.jpg']


@pytest.fixture
def empty_image(request):
    yield request.param


@pytest.fixture
def person_image(request):
    yield request.param


@pytest.fixture
def posenet_model(request):
    filepath = os.path.join(os.getcwd(), 'tests', 'pipeline', 'nodes',
                            'model', 'posenetv1', 'test_posenet.yml')
    with open(filepath) as file:
        node_config = yaml.safe_load(file)
    node_config['root'] = os.getcwd()
    node_config['model_type'] = request.param
    node = Node(node_config)
    yield node

@pytest.mark.mlmodel
class TestPoseNet:
    @pytest.mark.parametrize('empty_image', empty_image_list, indirect=True, ids=str)
    @pytest.mark.parametrize('posenet_model', models, indirect=True, ids=str)
    def test_no_detection(self, posenet_model, empty_image):
        blank_image = cv2.imread(os.path.join(TEST_DIR, empty_image))
        assert posenet_model is not None
        output = posenet_model.run({'img': blank_image})
        expected_output = {"bboxes": np.zeros(0),
                           "keypoints": np.zeros(0),
                           "keypoint_scores": np.zeros(0),
                           "keypoint_conns": np.zeros(0),
                           "bbox_labels": np.zeros(0)}
        assert output.keys() == expected_output.keys(), "missing keys"
        for i in expected_output.keys():
            npt.assert_array_equal(output[i], expected_output[i]), "unexpected output for {}".format(i)

    @pytest.mark.parametrize('posenet_model', models, indirect=True, ids=str)
    @pytest.mark.parametrize('person_image', person_image_list, indirect=True, ids=str)
    def test_different_models(self, posenet_model, person_image):
        pose_image = cv2.imread(os.path.join(TEST_DIR, person_image))
        assert posenet_model is not None
        output = posenet_model.run({'img': pose_image})
        expected_output = dict.fromkeys(['bboxes', 'keypoints', 'keypoint_scores',
                                         'keypoint_conns', 'bbox_labels'])
        assert output.keys() == expected_output.keys(), "missing keys"
        for i in expected_output.keys():
            assert len(output[i]) >= 1, "unexpected number of outputs for {}".format(i)
        for label in output['bbox_labels']:
            assert label == 'Person'
