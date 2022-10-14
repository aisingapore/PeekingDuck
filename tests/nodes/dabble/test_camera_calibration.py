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

"""
Test for dabble camera calibration node
"""

import gc

import cv2
import numpy as np
import pytest
import tensorflow.keras.backend as K

from peekingduck.nodes.dabble.camera_calibration import (
    Node,
    _check_corners_validity,
    _get_box_info,
)
from tests.conftest import TEST_DATA_DIR, TEST_IMAGES_DIR, not_raises

CORNER_DATA = ["corners_ok.npz", "image_too_small.npz", "not_in_box.npz"]
CHECKERBOARD_IMAGES = ["checkerboard1.png", "checkerboard2.png"]
NO_CHECKERBOARD_IMAGES = ["black.jpg"]
DETECTED_CORNERS = ["detected_corners.npz"]
CALIBRATION_DATA = ["calibration_data.npz"]


@pytest.fixture
def camera_calibration_node():
    node = Node(
        {
            "input": ["img"],
            "output": ["img"],
            "num_corners": [10, 7],
            "scale_factor": 2,
            "file_path": "PeekingDuck/data/camera_calibration_coeffs.yml",
        }
    )
    return node


@pytest.fixture(params=CORNER_DATA)
def corner_data(request):
    yield str(TEST_DATA_DIR / "camera_calibration" / request.param)
    K.clear_session()
    gc.collect()


@pytest.fixture(params=CHECKERBOARD_IMAGES)
def checkerboard_images(request):
    yield str(TEST_IMAGES_DIR / request.param)
    K.clear_session()
    gc.collect()


@pytest.fixture(params=NO_CHECKERBOARD_IMAGES)
def no_checkerboard_images(request):
    yield str(TEST_IMAGES_DIR / request.param)
    K.clear_session()
    gc.collect()


@pytest.fixture(params=DETECTED_CORNERS)
def detected_corners(request):
    yield str(TEST_DATA_DIR / "camera_calibration" / request.param)
    K.clear_session()
    gc.collect()


@pytest.fixture(params=CALIBRATION_DATA)
def calibration_data(request):
    yield str(TEST_DATA_DIR / "camera_calibration" / request.param)
    K.clear_session()
    gc.collect()


@pytest.mark.usefixtures("tmp_dir")
class TestCameraCalibration:
    def test_file_io(self):
        with pytest.raises(ValueError) as excinfo:
            Node(
                {
                    "input": ["img"],
                    "output": ["img"],
                    "num_corners": [10, 7],
                    "scale_factor": 4,
                    "file_path": "file.txt",
                }
            )
        assert str(excinfo.value) == "Filepath must have a '.yml' extension."

    def test_check_corner_validity(self, corner_data):
        file_path = str(corner_data)

        data = np.load(file_path)
        width = data["width"]
        height = data["height"]
        corners = data["corners"]
        start_point = data["start_point"]
        end_point = data["end_point"]
        expected_output = data["expected_output"]

        assert (
            _check_corners_validity(width, height, corners, start_point, end_point)
            == expected_output
        )

    def test_get_box_info(self):
        for i in range(5):
            start_points, end_points, (text_pos, pos_type) = _get_box_info(i, 1280, 720)
            assert all(isinstance(val, int) for val in start_points)
            assert all(isinstance(val, int) for val in end_points)
            assert all(isinstance(val, int) for val in text_pos)
            assert isinstance(pos_type, int)

    def test_check_initialize_display_scales(self, camera_calibration_node):
        camera_calibration_node._initialize_display_scales(1280)
        assert camera_calibration_node.display_scales

    def test_checkerboard_detection(self, camera_calibration_node, checkerboard_images):
        img = cv2.imread(checkerboard_images)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img.shape[:2]

        success, corners = camera_calibration_node._detect_corners(
            height, width, gray_img
        )
        assert success
        assert corners.shape == (70, 1, 2)

    def test_no_checkerboard_detection(
        self, camera_calibration_node, no_checkerboard_images
    ):
        img = cv2.imread(no_checkerboard_images)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img.shape[:2]

        success, corners = camera_calibration_node._detect_corners(
            height, width, gray_img
        )
        assert not success

    def test_calculate_coeffs(self, camera_calibration_node, detected_corners):
        data = np.load(detected_corners)
        camera_calibration_node.object_points = data["object_points"]
        camera_calibration_node.image_points = data["image_points"]

        calibration_data = camera_calibration_node._calculate_coeffs(data["img_shape"])
        (
            calibration_success,
            camera_matrix,
            distortion_coeffs,
            _,
            _,
        ) = calibration_data
        assert calibration_success
        assert camera_matrix.shape == (3, 3)
        assert distortion_coeffs.shape == (1, 5)

    def test_calculate_error(
        self, camera_calibration_node, detected_corners, calibration_data
    ):
        data = np.load(detected_corners)
        camera_calibration_node.object_points = data["object_points"]
        camera_calibration_node.image_points = data["image_points"]

        calibration_data = np.load(calibration_data, allow_pickle=True)[
            "calibration_data"
        ]

        with not_raises(Exception):
            camera_calibration_node._calculate_error(calibration_data)

    def test_run(self, camera_calibration_node, checkerboard_images):
        img = cv2.imread(checkerboard_images)

        with not_raises(Exception):
            camera_calibration_node.run({"img": img})
