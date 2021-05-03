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

SIZE = (900, 800, 3)


@pytest.fixture
def image():
    res = np.random.randint(255, size=SIZE, dtype=np.uint8)
    return res


@pytest.fixture
def images():
    def generate_img():
        return np.random.randint(255, size=SIZE, dtype=np.uint8)
    res = [generate_img() for _ in range(30)]
    return res


@pytest.fixture
def create_image():

    def _create_image(size):
        img = np.random.randint(255, size=size, dtype=np.uint8)
        return img

    return _create_image


@pytest.fixture
def create_video():

    def _create_video(size):
        res = [np.random.randint(255, size=size, dtype=np.uint8)
               for i in range(30)]
        return res

    return _create_video


@pytest.fixture
def create_input_video(create_video):

    def _create_input_image(path, size):
        vid = create_video(size)
        cv2.imwrite(path, vid)
        return vid

    return _create_input_image


@pytest.fixture
def create_input_image(create_image):

    def _create_input_image(path, size):
        img = create_image(size)
        cv2.imwrite(path, img)
        return img

    return _create_input_image


@pytest.fixture
def tmp_dir():
    cwd = os.getcwd()
    newpath = tempfile.mkdtemp()
    os.chdir(newpath)
    yield
    os.chdir(cwd)
    shutil.rmtree(newpath)
