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

import re
from pathlib import Path

import pytest

from peekingduck.pipeline.nodes.output.media_writer import Node

OUTPUT_PATH = Path("output")
SIZE = (400, 600, 3)


def directory_contents():
    return list(set(OUTPUT_PATH.iterdir()))


@pytest.fixture
def writer():
    media_writer = Node({"output_dir": OUTPUT_PATH, "input": "img", "output": "none"})
    return media_writer


@pytest.mark.usefixtures("tmp_dir")
class TestMediaWriter:
    def test_cwd_starts_empty(self):
        assert list(Path.cwd().iterdir()) == []

    def test_writer_writes_single_image(self, writer, create_image):
        image = create_image(SIZE)
        writer.run(
            {
                "filename": "test.jpg",
                "img": image,
                "saved_video_fps": 1,
                "pipeline_end": False,
            }
        )
        writer.run(
            {
                "filename": "test.jpg",
                "img": None,
                "saved_video_fps": 1,
                "pipeline_end": True,
            }
        )

        # pattern to check for time stamp filename_DDMMYY-hh-mm-ss.extension
        # approved extension = ["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"]
        # listed in input.recorded.py
        pattern = r".*_\d{6}-\d{2}-\d{2}-\d{2}\.[a-z0-9]{3,4}$"

        assert len(directory_contents()) == 1
        assert directory_contents()[0].suffix == ".jpg"
        assert re.search(pattern, str(directory_contents()[0]))

    def test_writer_writes_multi_image(self, writer, create_image):
        image1 = create_image(SIZE)
        image2 = create_image(SIZE)
        image3 = create_image(SIZE)

        writer.run(
            {
                "filename": "test1.jpg",
                "img": image1,
                "saved_video_fps": 1,
                "pipeline_end": False,
            }
        )
        writer.run(
            {
                "filename": "test2.jpg",
                "img": image2,
                "saved_video_fps": 1,
                "pipeline_end": False,
            }
        )
        writer.run(
            {
                "filename": "test3.jpg",
                "img": image3,
                "saved_video_fps": 1,
                "pipeline_end": False,
            }
        )
        writer.run(
            {
                "filename": "test3.jpg",
                "img": None,
                "saved_video_fps": 1,
                "pipeline_end": True,
            }
        )

        assert len(directory_contents()) == 3

        # pattern to check for time stamp filename_DDMMYY-hh-mm-ss.extension
        # approved extension = ["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"]
        # listed in input.recorded.py
        pattern = r".*_\d{6}-\d{2}-\d{2}-\d{2}\.[a-z0-9]{3,4}$"

        for filename in directory_contents():
            assert filename.suffix == ".jpg"
            assert re.search(pattern, str(filename))

    def test_writer_writes_single_video(self, writer, create_video):
        video = create_video(SIZE, nframes=20)
        for frame in video:
            writer.run(
                {
                    "filename": "test.mp4",
                    "img": frame,
                    "saved_video_fps": 30,
                    "pipeline_end": False,
                }
            )
        writer.run(
            {
                "filename": "test.mp4",
                "img": None,
                "saved_video_fps": 30,
                "pipeline_end": True,
            }
        )
        # pattern to check for time stamp filename_DDMMYY-hh-mm-ss.extension
        # approved extension = ["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"]
        # listed in input.recorded.py
        pattern = r".*_\d{6}-\d{2}-\d{2}-\d{2}\.[a-z0-9]{3,4}$"

        assert len(directory_contents()) == 1
        assert directory_contents()[0].suffix == ".mp4"
        assert re.search(pattern, str(directory_contents()[0]))

    def test_writer_writes_multi_video(self, writer, create_video):
        video1 = create_video(SIZE, nframes=20)
        video2 = create_video(SIZE, nframes=20)
        for frame in video1:
            writer.run(
                {
                    "filename": "test1.mp4",
                    "img": frame,
                    "saved_video_fps": 10,
                    "pipeline_end": False,
                }
            )
        for frame in video2:
            writer.run(
                {
                    "filename": "test2.mp4",
                    "img": frame,
                    "saved_video_fps": 10,
                    "pipeline_end": False,
                }
            )
        writer.run(
            {
                "filename": "test2.mp4",
                "img": None,
                "saved_video_fps": 10,
                "pipeline_end": True,
            }
        )
        assert len(directory_contents()) == 2

        # pattern to check for time stamp filename_DDMMYY-hh-mm-ss.extension
        # approved extension = ["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"]
        # listed in input.recorded.py
        pattern = r".*_\d{6}-\d{2}-\d{2}-\d{2}\.[a-z0-9]{3,4}$"

        for filename in directory_contents():
            assert filename.suffix == ".mp4"
            assert re.search(pattern, str(filename))
