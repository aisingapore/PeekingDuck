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
Calculates camera coefficients for undistortion
"""

from typing import Any, Dict, List

from pathlib import Path
import math
import time
import cv2
import yaml
import numpy as np

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.draw.utils.constants import CHAMPAGNE, BLACK, TOMATO

# global constants
# terminal criteria for subpixel finetuning
TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
TOP_LEFT = 0
TOP_RIGHT = 1
BOTTOM_LEFT = 2
BOTTOM_RIGHT = 3
MIDDLE = 4
CORNERS_OK = 0
IMAGE_TOO_SMALL = 1
NOT_IN_BOX = 2
DEFAULT_TEXT = "PLACE BOARD HERE"
TOO_SMALL = "MOVE BOARD CLOSER"
DETECTION_SUCCESS = "DETECTION SUCCESSFUL! PRESS ANY KEY TO CONTINUE."
DETECTION_COMPLETE = "DETECTION COMPLETE! PRESS ANY KEY TO EXIT."
BOX_OPACITY = 0.75


def get_box_info(num: int, width: int, height: int) -> tuple:
    """Gets start and end points of box, and position to put text"""
    start_points = {
        TOP_LEFT: (0, 0),
        TOP_RIGHT: (int(2 * width / 3), 0),
        BOTTOM_LEFT: (0, int(height / 2)),
        BOTTOM_RIGHT: (int(2 * width / 3), int(height / 2)),
        MIDDLE: (int(width / 3), int(height / 4)),
    }
    end_points = {
        TOP_LEFT: (int(width / 3), int(height / 2)),
        TOP_RIGHT: (width, int(height / 2)),
        BOTTOM_LEFT: (int(width / 3), height),
        BOTTOM_RIGHT: (width, height),
        MIDDLE: (int(2 * width / 3), int(3 * height / 4)),
    }
    text_positions = {
        TOP_LEFT: (5, int(height / 2) - 5),
        TOP_RIGHT: (int(2 * width / 3) + 5, int(height / 2) - 5),
        BOTTOM_LEFT: (5, int(height / 2) + 5),
        BOTTOM_RIGHT: (int(2 * width / 3) + 5, int(height / 2) + 5),
        MIDDLE: (int(width / 3) + 5, int(3 * height / 4) - 5),
    }
    pos_types = {
        TOP_LEFT: BOTTOM_LEFT,
        TOP_RIGHT: BOTTOM_LEFT,
        BOTTOM_LEFT: TOP_LEFT,
        BOTTOM_RIGHT: TOP_LEFT,
        MIDDLE: BOTTOM_LEFT,
    }

    return start_points[num], end_points[num], (text_positions[num], pos_types[num])


def get_optimal_font_scale(text: str, width: int) -> float:
    """Helper function to get optimal font scale given text and width"""
    for scale in range(250, 25, -1):
        thickness = int((scale / 50) / 0.5)

        text_size = cv2.getTextSize(
            text=text,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=scale / 50,
            thickness=thickness,
        )
        new_width = text_size[0][0]
        if new_width <= width:
            return scale / 50
    return 0.5


def draw_bgnd_box(
    img: np.ndarray,
    pt1: tuple,
    pt2: tuple,
) -> np.ndarray:
    """Helper function to draw box on image"""

    box_img = img.copy()

    # draw the rectangle
    cv2.rectangle(box_img, pt1, pt2, BLACK, cv2.FILLED)

    # apply the overlay
    return cv2.addWeighted(box_img, BOX_OPACITY, img, 1 - BOX_OPACITY, 0, img)


def draw_text(img: np.ndarray, text: str, pos_info: tuple) -> np.ndarray:
    """Helper function to draw text on the image"""

    pos, pos_type = pos_info

    img_copy = img.copy()

    font_scale = get_optimal_font_scale(text, img_copy.shape[1] / 3 - 10)
    thickness = int(font_scale / 0.5)

    sentences = [""]
    for word in text.split():
        last_str = sentences[-1]
        if last_str == "":
            new_str = last_str + word
        else:
            new_str = last_str + " " + word

        text_width = cv2.getTextSize(
            text=new_str,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            thickness=thickness,
        )[0][0]
        if text_width > img_copy.shape[1] / 3 - 10:
            sentences.append(word)
        else:
            sentences[-1] = new_str

    text_width = 0
    text_height = cv2.getTextSize(
        text=sentences[0],
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        thickness=thickness,
    )[0][1]

    for sentence in sentences:
        text_width = max(
            text_width,
            cv2.getTextSize(
                text=sentence,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                thickness=thickness,
            )[0][0],
        )

    if pos_type == TOP_LEFT:
        img_copy = draw_bgnd_box(
            img_copy,
            (pos[0], pos[1]),
            (pos[0] + text_width + 5, pos[1] + len(sentences) * (text_height + 5) + 5),
        )

        pos = (pos[0] + 5, pos[1] + text_height + 5)

    elif pos_type == BOTTOM_LEFT:
        img_copy = draw_bgnd_box(
            img_copy,
            (pos[0], pos[1] - len(sentences) * (text_height + 5) - 5),
            (pos[0] + text_width + 5, pos[1]),
        )

        pos = (pos[0] + 5, pos[1] - (text_height + 5) * (len(sentences) - 1) - 5)

    for sentence in sentences:
        img_copy = cv2.putText(
            img=img_copy,
            text=sentence,
            org=pos,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=CHAMPAGNE,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        pos = (pos[0], pos[1] + text_height + 5)

    return img_copy


def draw_countdown(img: np.ndarray, num: int) -> np.ndarray:
    """Helper function to draw the countdown in the center of the screen"""

    img_copy = img.copy()
    height, width = img_copy.shape[:2]

    text = str(num)

    font_scale = get_optimal_font_scale(text, width / 8)
    text_width, text_height = cv2.getTextSize(
        text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=2
    )[0]

    pos = (int(width / 2 - text_width / 2), int(height / 2 + text_height / 2))

    cv2.putText(
        img=img_copy,
        text=text,
        org=pos,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=BLACK,
        thickness=10,
        lineType=cv2.LINE_AA,
    )

    return cv2.putText(
        img=img_copy,
        text=text,
        org=pos,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=TOMATO,
        thickness=8,
        lineType=cv2.LINE_AA,
    )


def draw_box(img: np.ndarray, start_point: tuple, end_point: tuple) -> np.ndarray:
    """Helper function to draw rectangle on the image"""

    img_copy = img.copy()

    cv2.rectangle(
        img=img_copy,
        pt1=start_point,
        pt2=end_point,
        color=(0, 0, 0),
        thickness=3,
    )

    return cv2.rectangle(
        img=img_copy,
        pt1=start_point,
        pt2=end_point,
        color=CHAMPAGNE,
        thickness=1,
    )


def check_corners_valid(
    width: int, height: int, corners: np.ndarray, start_point: tuple, end_point: tuple
) -> int:
    """Checks whether the corners are large enough and fall within the box"""

    min_w = width
    min_h = height
    max_w = 0
    max_h = 0
    for corner in corners:
        min_w = min(min_w, corner[0][0])
        max_w = max(max_w, corner[0][0])
        min_h = min(min_h, corner[0][1])
        max_h = max(max_h, corner[0][1])

    area = (max_w - min_w) * (max_h - min_h)

    if area < width * height / (6 * 4):
        return IMAGE_TOO_SMALL

    min_w_box = max(min_w, start_point[0])
    max_w_box = min(max_w, end_point[0])
    min_h_box = max(min_h, start_point[1])
    max_h_box = min(max_h, end_point[1])

    if max_w_box < min_w_box or max_h_box < min_h_box:
        return NOT_IN_BOX

    if (max_w_box - min_w_box) * (max_h_box - min_h_box) < 3 * area / 4:
        return NOT_IN_BOX

    return CORNERS_OK


class Node(AbstractNode):
    """Calculates camera coefficients for undistortion
    <https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html>'_.

    Inputs:
        |img_data|

    Outputs:
        |img_data|


    Configs:
        num_corners (:obj:'List[int]'):
            **default = [10, 7]**. |br|
            A list containing the number of internal corners
            along the vertical and horizontal.
        num_pictures (:obj:'int'):
            **default = 5**. |br|
            Number of pictures to take to calculate the coefficients
        scale_factor (:obj:'int'):
            **default = 4**. |br|
            How much to scale the image by when finding chessboard corners. For
            example, with a scale of 4, an image of size (1080 x 1920) will be
            scaled down to (270 x 480) when detecting the corners. Increasing this
            value reduces computation time. If the node is unable to detect corners,
            reducing this value may help.
        file_path (:obj:`str`):
            **default = "PeekingDuck/data/camera_calibration_coeffs.yml"**. |br|
            Path of the YML file to store the calculated camera coefficients.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.file_path = Path(self.file_path)  # type: ignore
        # check if file_path has a ".yml" extension
        if self.file_path.suffix != ".yml":
            raise ValueError("Filepath must have a '.yml' extension.")
        if not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)

        grid_height = self.num_corners[0]
        grid_width = self.num_corners[1]

        # prepare all object points, like (0, 0, 0), (1, 0, 0), etc.
        np_mgrid = np.mgrid[0:grid_height, 0:grid_width]
        self.object_points_base = np.zeros((grid_height * grid_width, 3), np.float32)
        self.object_points_base[:, :2] = np_mgrid.T.reshape(-1, 2)

        # arrays to store object points and image points
        self.object_points: List[np.ndarray] = []  # points in real world
        self.image_points: List[np.ndarray] = []  # points on image plane

        self.last_detection = time.time()
        self.num_detections = 0

        self.num_pictures = 5

    def detect_corners(self, height: int, width: int, gray_img: np.ndarray) -> tuple:
        """Tries to detect corners in the image"""
        # downscale
        new_h = int(height / self.scale_factor)
        new_w = int(width / self.scale_factor)

        resized_img = cv2.resize(gray_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # try to find chessboard corners
        return cv2.findChessboardCorners(
            image=resized_img, patternSize=self.num_corners, corners=None
        )

    def calculate_coeffs(self, img_shape: tuple) -> None:
        """Performs calculations with detected corners"""
        (
            calibration_success,
            camera_matrix,
            distortion_coeffs,
            rotation_vec,
            translation_vec,
        ) = cv2.calibrateCamera(
            objectPoints=self.object_points,
            imagePoints=self.image_points,
            imageSize=img_shape,
            cameraMatrix=None,
            distCoeffs=None,
        )

        if calibration_success:
            self.logger.info("Calibration successful!")
            self.logger.info(f"camera_matrix: {camera_matrix}")
            self.logger.info(f"distortion_coeffs: {distortion_coeffs}")

            file_data = {}
            file_data["camera_matrix"] = camera_matrix.tolist()
            file_data["distortion_coeffs"] = distortion_coeffs.tolist()

            yaml.dump(file_data, open(self.file_path, "w"), default_flow_style=None)
        else:
            raise Exception("Calibration failed. Please try again.")

        mean_error = 0
        for i in range(len(self.object_points)):
            projected_image_points, _ = cv2.projectPoints(
                objectPoints=self.object_points[i],
                rvec=rotation_vec[i],
                tvec=translation_vec[i],
                cameraMatrix=camera_matrix,
                distCoeffs=distortion_coeffs,
            )

            error = cv2.norm(
                src1=self.image_points[i],
                src2=projected_image_points,
                normType=cv2.NORM_L2,
            )
            error /= len(projected_image_points)
            mean_error += error

        self.logger.info(f"total eror: {mean_error / len(self.object_points)}")

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node calculates the camera distortion coefficients for undistortion.

        Args:
            inputs (dict): Inputs dictionary with the key `img`.

        Returns:
            outputs (dict): Outputs dictionary with the key `img`.
        """

        img = inputs["img"]
        img = cv2.flip(img, 1)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detect_corners_success = False

        height, width = img.shape[:2]

        start_point, end_point, text_pos = get_box_info(
            self.num_detections, width, height
        )

        img_notext = draw_box(img, start_point, end_point)
        img = draw_text(
            img_notext,
            f"{DEFAULT_TEXT} ({self.num_detections+1}/{self.num_pictures})",
            text_pos,
        )

        # if sufficient time has passed
        if time.time() - self.last_detection >= 5:
            detect_corners_success, corners = self.detect_corners(
                height, width, gray_img
            )

            if corners is not None:
                corners = corners * self.scale_factor

        # cv2 successfully detected the corners
        if detect_corners_success:
            corners_valid = check_corners_valid(
                width, height, corners, start_point, end_point
            )

            if corners_valid != CORNERS_OK:
                detect_corners_success = False
                if corners_valid == IMAGE_TOO_SMALL:
                    img = draw_text(img_notext, TOO_SMALL, text_pos)

        if detect_corners_success:

            self.object_points.append(self.object_points_base)
            self.image_points.append(corners)

            # improve corner accuracy
            corners_accurate = cv2.cornerSubPix(
                image=gray_img,
                corners=corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=TERMINATION_CRITERIA,
            )

            # draw corners and message on the image
            cv2.drawChessboardCorners(
                image=img_notext,
                patternSize=self.num_corners,
                corners=corners_accurate,
                patternWasFound=detect_corners_success,
            )

            self.num_detections += 1

            if self.num_detections != self.num_pictures:
                img = draw_text(img_notext, DETECTION_SUCCESS, text_pos)
            else:
                img = draw_text(img_notext, DETECTION_COMPLETE, text_pos)

            # display the image and wait for user to press a key
            cv2.imshow("PeekingDuck", img)
            cv2.waitKey(0)

            self.last_detection = time.time()

            # if we have sufficient images, calculate the coefficients and write to a file
            if self.num_detections == self.num_pictures:
                self.calculate_coeffs(img_shape=gray_img.shape[::-1])
                return {"pipeline_end": True}

        time_to_next_detection = math.ceil(5 - time.time() + self.last_detection)
        if time_to_next_detection > 0:
            img = draw_countdown(img, time_to_next_detection)

        return {"img": img}
