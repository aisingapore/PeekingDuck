# Copyright 2021 AI Singapore
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

import os
import shutil
import tempfile

import numpy as np
import pytest

import cv2


@pytest.fixture
def create_image():

    def _create_image(size):
        img = np.random.randint(255, size=size, dtype=np.uint8)
        return img

    return _create_image


@pytest.fixture
def create_input_image(create_image):

    def _create_input_image(path, size):
        img = create_image(size)
        cv2.imwrite(path, img)
        return img

    return _create_input_image


@pytest.fixture
def create_video():

    def _create_video(size, nframes):
        res = [np.random.randint(255, size=size, dtype=np.uint8)
               for _ in range(nframes)]
        return res

    return _create_video


@pytest.fixture
def create_input_video(create_video):

    def _create_input_video(path, fps, size, nframes):
        vid = create_video(size, nframes)
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')
        resolution = (size[1], size[0])
        writer = cv2.VideoWriter(path, fourcc, fps, resolution)
        for frame in vid:
            writer.write(frame)
        return vid

    return _create_input_video


@pytest.fixture
def tmp_dir():
    cwd = os.getcwd()
    newpath = tempfile.mkdtemp()
    os.chdir(newpath)
    yield
    os.chdir(cwd)
    shutil.rmtree(newpath)


@pytest.fixture
def root_dir():
    rootdir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'peekingduck'
    )
    return rootdir


@pytest.fixture
def test_human_images(root_dir):
    test_dir = os.path.join(root_dir, '..', 'tests')
    test_img_dir = os.path.join(test_dir, 'test_images', 'human')
    TEST_IMAGES_NAMES = os.listdir(test_img_dir)

    test_img_paths = [os.path.join(test_img_dir, img_name) for img_name in TEST_IMAGES_NAMES]

    return test_img_paths


@pytest.fixture
def test_black_image(root_dir):
    test_dir = os.path.join(root_dir, '..', 'tests')
    test_img_dir = os.path.join(test_dir, 'test_images')
    black_img_path = os.path.join(test_img_dir, 'black.jpeg')

    return black_img_path


@pytest.fixture
def test_animal_image(root_dir):
    test_dir = os.path.join(root_dir, '..', 'tests')
    test_img_dir = os.path.join(test_dir, 'test_images')
    black_img_path = os.path.join(test_img_dir, 't3.jpg')

    return black_img_path
