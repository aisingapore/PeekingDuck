# Copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager

import numpy as np
import pytest
from unittest import TestCase

from peekingduck.pipeline.nodes.input.visual import Node


@contextmanager
def not_raises(exception):
    try:
        yield
    except exception:
        raise pytest.fail(f"DID RAISE EXCEPTION: {exception}")


def create_reader(source=None):
    media_reader = Node(
        {
            "input": "source",
            "output": "img",
            "resize": {"do_resizing": False, "width": 1280, "height": 720},
            "filename": "video.mp4",
            "frames_log_freq": 100,
            "mirror_image": False,
            "pipeline_end": False,
            "saved_video_fps": 0,
            "threading": False,
            "source": source if source else ".",
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
                    "source": file_path,
                }
            )

    def test_reader_run_fine_on_empty_folder(self):
        with not_raises(FileNotFoundError) as excinfo:
            reader = create_reader()
            reader.run({})

    def test_reader_reads_one_image(self, create_input_image):
        filename = "image1.png"
        image1 = create_input_image(filename, (900, 800, 3))
        reader = create_reader(source=filename)
        output1 = reader.run({})
        assert np.array_equal(output1["img"], image1)
        assert output1["filename"] == filename

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
        video1 = create_input_video(
            "video1.avi", fps=10, size=size, num_frames=num_frames
        )
        reader = create_reader()

        read_video1 = _get_video_file(reader, num_frames)
        assert np.array_equal(read_video1, video1)

    def test_reader_reads_multiple_videos(self, create_input_video):
        num_frames = 20
        size = (600, 800, 3)

        video1 = create_input_video(
            "video1.avi", fps=5, size=size, num_frames=num_frames
        )
        video2 = create_input_video(
            "video2.avi", fps=5, size=size, num_frames=num_frames
        )

        reader = create_reader()

        read_video1 = _get_video_file(reader, num_frames)
        assert np.array_equal(read_video1, video1)

        read_video2 = _get_video_file(reader, num_frames)
        assert np.array_equal(read_video2, video2)

    def test_input_folder_of_mixed_media(self, create_input_image, create_input_video):
        """Test read a folder of mixed media files: images and videos, and verifying
        progress log messages
        """
        size = (640, 480, 3)
        test_filenames = [
            "test_image_1.jpg",
            "test_image_2.png",
            "test_image_3.png",
            "test_video_1.avi",
            "test_video_2.avi",
        ]
        contents = {
            "img1": create_input_image(test_filenames[0], size),
            "img2": create_input_image(test_filenames[1], size),
            "img3": create_input_image(test_filenames[2], size),
            # NB: be sure to sync number of frames with 'key_frames'!
            "vid_30": create_input_video(
                test_filenames[3], fps=10, num_frames=30, size=size
            ),
            "vid_3": create_input_video(
                test_filenames[4], fps=1, num_frames=3, size=size
            ),
        }
        msg_set = set()
        with TestCase.assertLogs("peekingduck.pipeline.nodes.input.visual") as captured:
            reader = create_reader()
            for k, v in contents.items():
                if k.startswith("vid"):
                    toks = k.split("_")  # decode number of frames
                    num_frames = int(toks[1])
                    print(f"num_frames={num_frames}")
                    for _ in range(num_frames):
                        reader.run({})
                else:
                    print(f"read {k}")
                    reader.run({})
            reader.run({})  # run last pipeline iteration

            for record in captured.records:
                msg = record.getMessage()
                msg_set.add(msg)

        print(msg_set)
        assert "Approximate Progress: 33%" in msg_set
        assert "Approximate Progress: 100%" in msg_set
        assert f"Completed processing file: {test_filenames[0]} (1 / 5)" in msg_set
        assert f"Completed processing file: {test_filenames[2]} (3 / 5)" in msg_set
        assert f"Completed processing file: {test_filenames[4]} (5 / 5)" in msg_set
