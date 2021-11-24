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

import numpy as np
import pytest

from peekingduck.pipeline.nodes.input.recorded import Node


def create_reader():
    media_reader = Node(
        {
            "input": "source",
            "output": "img",
            "resize": {"do_resizing": False, "width": 1280, "height": 720},
            "mirror_image": False,
            "threading": False,
            "input_dir": ".",
        }
    )
    return media_reader


def _get_video_file(reader, num_frames):
    """Helper function to get an entire videofile"""
    video = []
    for _ in range(num_frames):
        output = reader.run({})
        video.append(output["img"])
    return video


@pytest.mark.usefixtures("tmp_dir")
class TestMediaReader:
    def test_reader_run_throws_error_on_wrong_file_path(self):
        with pytest.raises(FileNotFoundError):
            file_path = "path_that_does_not_exist"
            Node(
                {
                    "input": "source",
                    "output": "img",
                    "resize": {"do_resizing": False, "width": 1280, "height": 720},
                    "mirror_image": False,
                    "input_dir": file_path,
                }
            )

    def test_reader_run_throws_error_on_empty_folder(self):
        with pytest.raises(FileNotFoundError):
            reader = create_reader()
            reader.run({})

    def test_reader_reads_one_image(self, create_input_image):
        image1 = create_input_image("image1.png", (900, 800, 3))
        reader = create_reader()
        output1 = reader.run({})
        assert np.array_equal(output1["img"], image1)

    def test_reader_reads_multi_images(self, create_input_image):
        image1 = create_input_image("image1.png", (900, 800, 3))
        image2 = create_input_image("image2.png", (900, 800, 3))
        image3 = create_input_image("image3.png", (900, 800, 3))
        reader = create_reader()
        output1 = reader.run({})
        output2 = reader.run({})
        output3 = reader.run({})

        assert np.array_equal(output1["img"], image1)
        assert np.array_equal(output2["img"], image2)
        assert np.array_equal(output3["img"], image3)

    def test_reader_reads_one_video(self, create_input_video):
        num_frames = 30
        size = (600, 800, 3)
        video1 = create_input_video("video1.avi", fps=10, size=size, nframes=num_frames)
        reader = create_reader()

        read_video1 = _get_video_file(reader, num_frames)
        assert np.array_equal(read_video1, video1)

    def test_reader_reads_multiple_videos(self, create_input_video):
        num_frames = 20
        size = (600, 800, 3)

        video1 = create_input_video("video1.avi", fps=5, size=size, nframes=num_frames)
        video2 = create_input_video("video2.avi", fps=5, size=size, nframes=num_frames)

        reader = create_reader()

        read_video1 = _get_video_file(reader, num_frames)
        assert np.array_equal(read_video1, video1)

        read_video2 = _get_video_file(reader, num_frames)
        assert np.array_equal(read_video2, video2)
