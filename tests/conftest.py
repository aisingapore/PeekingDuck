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
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import pytest
import tensorflow.keras.backend as K
import yaml

HUMAN_IMAGES = ["t1.jpg", "t2.jpg", "t4.jpg"]
NO_HUMAN_IMAGES = ["black.jpg", "t3.jpg"]

SINGLE_PERSON_IMAGES = ["t2.jpg"]
MULTI_PERSON_IMAGES = ["t1.jpg", "t4.jpg"]

HUMAN_VIDEOS = ["humans_mot.mp4"]
# Folders of frames from the video sequence
HUMAN_VIDEO_SEQUENCES = ["two_people_crossing"]

LICENSE_PLATE_IMAGES = ["tcar1.jpg", "tcar3.jpg", "tcar4.jpg"]
NO_LICENSE_PLATE_IMAGES = ["black.jpg", "t3.jpg"]

CROWD_IMAGES = ["crowd1.jpg", "crowd2.jpg"]

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
    def _create_video(size, num_frames):
        res = [
            np.random.randint(255, size=size, dtype=np.uint8) for _ in range(num_frames)
        ]
        return res

    return _create_video


@pytest.fixture
def create_input_video(create_video):
    def _create_input_video(path, fps, size, num_frames):
        vid = create_video(size, num_frames)
        fourcc = cv2.VideoWriter_fourcc(*"FFV1")
        resolution = (size[1], size[0])
        writer = cv2.VideoWriter(path, fourcc, fps, resolution)
        for frame in vid:
            writer.write(frame)
        return vid

    return _create_input_video


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


@pytest.fixture(params=HUMAN_IMAGES)
def human_image(request):
    yield str(TEST_IMAGES_DIR / request.param)
    K.clear_session()
    gc.collect()


@pytest.fixture(params=NO_HUMAN_IMAGES)
def no_human_image(request):
    yield str(TEST_IMAGES_DIR / request.param)
    K.clear_session()
    gc.collect()


@pytest.fixture(params=SINGLE_PERSON_IMAGES)
def single_person_image(request):
    yield str(TEST_IMAGES_DIR / request.param)
    K.clear_session()
    gc.collect()


@pytest.fixture(params=MULTI_PERSON_IMAGES)
def multi_person_image(request):
    yield str(TEST_IMAGES_DIR / request.param)
    K.clear_session()
    gc.collect()


@pytest.fixture(params=LICENSE_PLATE_IMAGES)
def license_plate_image(request):
    yield str(TEST_IMAGES_DIR / request.param)
    K.clear_session()
    gc.collect()


@pytest.fixture(params=NO_LICENSE_PLATE_IMAGES)
def no_license_plate_image(request):
    yield str(TEST_IMAGES_DIR / request.param)
    K.clear_session()
    gc.collect()


@pytest.fixture(params=CROWD_IMAGES)
def crowd_image(request):
    yield str(TEST_IMAGES_DIR / request.param)
    K.clear_session()
    gc.collect()


@pytest.fixture(params=HUMAN_VIDEOS)
def human_video(request):
    yield str(TEST_IMAGES_DIR / request.param)
    K.clear_session()
    gc.collect()


@pytest.fixture(params=HUMAN_VIDEO_SEQUENCES)
def human_video_sequence(request):
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


@pytest.fixture(params=HUMAN_VIDEO_SEQUENCES)
def human_video_sequence_with_empty_frames(request):
    """This actually returns a list of dictionaries each containing:
    - A video frame
    - Bounding boxes

    Yielding bounding box allows us to test dabble.tracking without having to
    attach a object detector before it.

    Yielding a list of frames instead of a video file allows for better control
    of test data and frame specific manipulations to trigger certain code
    branches.

    Additionally, we overwrite frames 0 and 5 to be blank frames to test if the
    trackers can handle no detections.
    """
    sequence_dir = TEST_DATA_DIR / "video_sequences" / request.param
    with open(sequence_dir / "detections.yml") as infile:
        detections = yaml.safe_load(infile.read())
    sequence = [
        {
            "img": cv2.imread(str(sequence_dir / f"{key}.jpg")),
            "bboxes": np.array(val["bboxes"]),
        }
        for key, val in detections.items()
    ]
    sequence[0]["img"] = np.zeros_like(sequence[0]["img"])
    sequence[0]["bboxes"] = np.empty((0, 4))
    sequence[5]["img"] = np.zeros_like(sequence[5]["img"])
    sequence[5]["bboxes"] = np.empty((0, 4))
    # Yielding video sequence name as well in case there are specific things to
    # check for based on video content
    yield request.param, sequence

    K.clear_session()
    gc.collect()


def do_nothing(*_):
    """Does nothing. For use with ``mock.patch``."""


def get_groundtruth(test_file_path):
    assert isinstance(test_file_path, Path)
    with open(test_file_path.parent / "test_groundtruth.yml") as infile:
        return yaml.safe_load(infile)


@contextmanager
def not_raises(exception):
    try:
        yield
    except exception:
        raise pytest.fail(f"DID RAISE EXCEPTION: {exception}")


def assert_msg_in_logs(msg: str, msg_logs) -> None:
    """Helper method to assert that given message is in a list of logged messages.

    Args:
        msg (str): message to check for
        msg_logs (List[LogRecord]): log records containing messages
    """
    res = False
    for log_record in msg_logs:
        if msg in log_record.getMessage():
            res = True
            break
    assert res
