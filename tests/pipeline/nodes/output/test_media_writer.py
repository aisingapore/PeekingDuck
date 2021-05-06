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

import pytest
from peekingduck.pipeline.nodes.output.media_writer import Node

OUTPUT_PATH = "output"


def directory_contents():
    res = os.listdir(OUTPUT_PATH)
    return set(res)


@pytest.fixture
def writer():
    media_writer = Node({"outputdir": OUTPUT_PATH,
                         "input": "img",
                         "output": "end"
                         })
    return media_writer


size = (400, 600, 3)


@pytest.mark.usefixtures("tmpdir")
class TestMediaWriter:

    def test_cwd_starts_empty(self):
        assert os.listdir(os.getcwd()) == []

    def test_writer_writes_single_image(self, writer, create_image):
        image = create_image(size)
        writer.run({"filename": "test.jpg", "img": image, "fps": 1})
        assert directory_contents() == set(["test.jpg"])

    def test_writer_writes_multi_image(self, writer, create_image):
        image1 = create_image(size)
        image2 = create_image(size)
        image3 = create_image(size)
        writer.run({"filename": "test1.jpg", "img": image1, "fps": 1})
        writer.run({"filename": "test2.jpg", "img": image2, "fps": 1})
        writer.run({"filename": "test3.jpg", "img": image3, "fps": 1})

        assert directory_contents() == set(
            ["test1.jpg", "test2.jpg", "test3.jpg"])

    def test_writer_writes_single_video(self, writer, create_video):
        video = create_video(size, nframes=20)
        for frame in video:
            writer.run({"filename": "test.mp4", "img": frame, "fps": 30})
        assert directory_contents() == set(["test.mp4"])

    def test_writer_writes_multi_video(self, writer, create_video):
        video1 = create_video(size, nframes=20)
        video2 = create_video(size, nframes=20)
        for frame in video1:
            writer.run({"filename": "test1.mp4", "img": frame, "fps": 10})

        for frame in video2:
            writer.run({"filename": "test2.mp4", "img": frame, "fps": 10})

        assert directory_contents() == set(['test1.mp4', 'test2.mp4'])
