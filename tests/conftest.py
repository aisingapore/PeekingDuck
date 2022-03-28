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

import gc
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import tensorflow.keras.backend as K
import yaml

TEST_HUMAN_IMAGES = ["t1.jpg", "t2.jpg", "t4.jpg"]
TEST_NO_HUMAN_IMAGES = ["black.jpg", "t3.jpg"]
TEST_HUMAN_VIDEOS = ["humans_mot.mp4"]
# Folders of frames from the video sequence
TEST_HUMAN_VIDEO_SEQUENCES = ["two_people_crossing"]

TEST_NO_LP_IMAGES = ["black.jpg", "t3.jpg"]
TEST_LP_IMAGES = ["tcar1.jpg", "tcar3.jpg", "tcar4.jpg"]

TEST_CROWD_IMAGES = ["crowd1.jpg", "crowd2.jpg"]

# Paths
PKD_DIR = Path(__file__).resolve().parents[1] / "peekingduck"
TEST_DATA_DIR = PKD_DIR.parent / "tests" / "data"
TEST_IMAGES_DIR = TEST_DATA_DIR / "images"


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
        res = [
            np.random.randint(255, size=size, dtype=np.uint8) for _ in range(nframes)
        ]
        return res

    return _create_video


@pytest.fixture
def create_input_video(create_video):
    def _create_input_video(path, fps, size, nframes):
        vid = create_video(size, nframes)
        fourcc = cv2.VideoWriter_fourcc(*"FFV1")
        resolution = (size[1], size[0])
        writer = cv2.VideoWriter(path, fourcc, fps, resolution)
        for frame in vid:
            writer.write(frame)
        return vid

    return _create_input_video


@pytest.fixture
def replace_download_weights():
    def _replace_download_weights(*_):
        return False

    return _replace_download_weights


@pytest.fixture
def tmp_dir():
    cwd = Path.cwd()
    newpath = tempfile.mkdtemp()
    os.chdir(newpath)
    yield
    os.chdir(cwd)
    shutil.rmtree(newpath, ignore_errors=True)  # ignore_errors for windows development


@pytest.fixture
def tmp_project_dir():
    """To used after `tmp_dir` fixture to simulate that we're in a proper
    custom PKD project directory
    """
    cwd = Path.cwd()
    (cwd / "tmp_dir").mkdir(parents=True)
    os.chdir("tmp_dir")
    yield
    os.chdir(cwd)


@pytest.fixture(params=TEST_HUMAN_IMAGES)
def test_human_images(request):
    yield str(TEST_IMAGES_DIR / request.param)
    K.clear_session()
    gc.collect()


@pytest.fixture(params=TEST_NO_HUMAN_IMAGES)
def test_no_human_images(request):
    yield str(TEST_IMAGES_DIR / request.param)
    K.clear_session()
    gc.collect()


@pytest.fixture(params=TEST_LP_IMAGES)
def test_lp_images(request):
    yield str(TEST_IMAGES_DIR / request.param)
    K.clear_session()
    gc.collect()


@pytest.fixture(params=TEST_NO_LP_IMAGES)
def test_no_lp_images(request):
    yield str(TEST_IMAGES_DIR / request.param)
    K.clear_session()
    gc.collect()


@pytest.fixture(params=TEST_CROWD_IMAGES)
def test_crowd_images(request):
    yield str(TEST_IMAGES_DIR / request.param)
    K.clear_session()
    gc.collect()


@pytest.fixture(params=TEST_HUMAN_VIDEOS)
def test_human_videos(request):
    yield str(TEST_IMAGES_DIR / request.param)
    K.clear_session()
    gc.collect()


@pytest.fixture(params=TEST_HUMAN_VIDEO_SEQUENCES)
def test_human_video_sequences(request):
    """This actually returns a list of dictionaries each containing:
    - A video frame
    - Bounding boxes

    Yielding bounding box allows us to test dabble.tracking without having to
    attach a object detector before it.

    Yielding a list of frames instead of a video file allows for better control
    of test data and frame specific manipulations to trigger certain code
    branches.
    """
    sequence_dir = TEST_DATA_DIR / "video_sequences" / request.param
    with open(sequence_dir / "detections.yml") as infile:
        detections = yaml.safe_load(infile.read())
    # Yielding video sequence name as well in case there are specific things to
    # check for based on video content
    yield request.param, [
        {
            "img": cv2.imread(str(sequence_dir / f"{key}.jpg")),
            "bboxes": np.array(val["bboxes"]),
        }
        for key, val in detections.items()
    ]

    K.clear_session()
    gc.collect()
