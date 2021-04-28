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

# https://stackoverflow.com/questions/9757299/python-testing-an-abstract-base-class
# https://stackoverflow.com/questions/51737334/pytest-deleting-files-created-by-the-tested-function

import os
import shutil
import tempfile

import numpy as np
import pytest
from peekingduck.pipeline.nodes.output.media_writer import Node

OUTPUT_PATH = "output"
SIZE = (900, 800, 3)


def directory_contents():
    res = os.listdir(OUTPUT_PATH)
    return set(res)


@pytest.fixture
def tmpdir():
    cwd = os.getcwd()
    newpath = tempfile.mkdtemp()
    os.chdir(newpath)
    yield
    os.chdir(cwd)
    shutil.rmtree(newpath)


@pytest.fixture
def image():
    res = np.random.randint(255, size=SIZE, dtype=np.uint8)
    return res


@pytest.fixture
def images():
    def generate_img():
        return np.random.randint(255, size=SIZE, dtype=np.uint8)
    res = [generate_img() for _ in range(90)]
    return res


@pytest.fixture
def writer():
    media_writer = Node({"outputdir": OUTPUT_PATH,
                         "input": "img",
                         "output": "end"
                         })
    return media_writer


@pytest.mark.usefixtures("tmpdir")
class TestMediaWriter:

    def test_cwd_again_starts_empty(self):
        assert os.listdir(os.getcwd()) == []

    def test_writer_writes_single_image(self, writer, image):
        writer.run({"filename": "test.jpg", "img": image, "fps": 1})
        assert directory_contents() == set(["test.jpg"])

    # def test_writer_writes_multi_image(self, writer):
    #     # TODO fix this test
    #     # Note there is a bug with multi image
    #     image1 = create_image()
    #     image2 = create_image()
    #     image3 = create_image()
    #     writer.run({"filename": "test1.jpg", "img": image1, "fps": 1})
    #     writer.run({"filename": "test2.jpg", "img": image2, "fps": 1})
    #     writer.run({"filename": "test3.jpg", "img": image3, "fps": 1})
    #     assert os.listdir("tmp") == ["test1.jpg"]

    def test_writer_writes_single_video(self, writer, images):

        for image in images:
            writer.run({"filename": "test.mp4", "img": image, "fps": 30})
        assert directory_contents() == set(["test.mp4"])

    def test_writer_writes_multi_video(self, writer, images):
        for image in images:
            writer.run({"filename": "test1.mp4", "img": image, "fps": 30})

        for image in images:
            writer.run({"filename": "test2.mp4", "img": image, "fps": 30})

        assert directory_contents() == set(['test1.mp4', 'test2.mp4'])
