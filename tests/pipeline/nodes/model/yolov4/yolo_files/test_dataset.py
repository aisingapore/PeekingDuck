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

import pytest
import numpy as np
from tensorflow import Tensor
from peekingduck.pipeline.nodes.model.yolov4.yolo_files.dataset import transform_images


@pytest.fixture
def large_image_array():
    return Tensor(np.ones((1280, 720, 3)))

@pytest.fixture
def small_image_array():
    return Tensor(np.ones((1, 1, 3)))

def test_transform_images_larger_images(large_image_array):
    transformed_img = transform_images(large_image_array, 416)
    assert transformed_img.shape == (416, 416, 3)
    assert transformed_img.dtype == np.float64

    all_elements_combinations = zip(range(416), range(416), range(3))
    assert [transformed_img[i][j][k] == 1/255 for i, j, k in all_elements_combinations]

def test_transform_images_smaller_images(small_image_array):
    transformed_img = transform_images(large_image_array, 416)
    assert transformed_img.shape == (416, 416, 3)
    assert transformed_img.dtype == np.float64
    assert transformed_img[0][0][0] == 1/225

    all_elements_combinations = zip(range(1, 416), range(1, 416), range(3))
    assert [transformed_img[i][j][k] == 0 for i, j, k in all_elements_combinations]