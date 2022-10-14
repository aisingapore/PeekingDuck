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
Calculates camera coefficients to be used to
remove distortion from a wide-angle camera image.
"""

import math
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import yaml

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.draw.utils.constants import BLACK, CHAMPAGNE, TOMATO

# global constants
# terminal criteria for subpixel finetuning
TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

NUM_PICTURES = 5

# constants for positioning of boxes and text
TOP_LEFT = 0
TOP_RIGHT = 1
BOTTOM_LEFT = 2
BOTTOM_RIGHT = 3
MIDDLE = 4

# constants for _check_corners_validity
AREA_THRESHOLD = 3 / 4
CORNERS_OK = 0
IMAGE_TOO_SMALL = 1
NOT_IN_BOX = 2

# displayed messages
DEFAULT_TEXT = ["PLACE BOARD HERE"]
TOO_SMALL = ["MOVE BOARD CLOSER"]
DETECTION_SUCCESS = ["DETECTION SUCCESSFUL!", "PRESS ANY KEY TO CONTINUE."]
DETECTION_COMPLETE = ["DETECTION COMPLETE!", "PRESS ANY KEY TO EXIT."]
MAX_LEN_TEXT = "PRESS ANY KEY TO CONTINUE."

# constants for drawing
BGND_BOX_OPACITY = 0.75
BOX_WIDTH_RATIO = 1 / 3  # relative to the window width
BOX_HEIGHT_RATIO = 1 / 2  # relative to the window height
TEXT_PADDING = 5

# constants for font drawing
# 1280 is used as the reference point for the ratio
BOX_THICKNESS_RATIO = 2 / 1280
NORMAL_FONT_SCALE_RATIO = 0.84 / 1280
COUNTDOWN_FONT_SCALE_RATIO = 10 / 1280
NORMAL_FONT_THICKNESS_RATIO = 2 / 1280
COUNTDOWN_FONT_THICKNESS_RATIO = 8 / 1280


class Node(AbstractNode):
    """Calculates camera coefficients for `undistortion
    <https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html>`_.

    To calculate your camera, first download the following checkerboard and print it
    out in a suitable size and attach it to a hard surface, or display it on a sufficiently
    large device screen, such as a computer or a tablet. For most use cases, an A4-sized
    checkerboard works well, but depending on the position and distance of the camera, a
    bigger checkerboard may be required.

    .. image:: /assets/api/checkerboard.png
        :width: 20 %

    Next, create an empty ``pipeline_config.yml`` in your project folder and modify it as follows:

    .. code-block:: yaml
        :linenos:

        nodes:
        - input.visual:
            source: 0 # change this to the camera you are using
            threading: True
            mirror_image: True
        - dabble.camera_calibration
        - output.screen

    Run the above pipeline with :greenbox:`peekingduck run`. If you are unfamiliar with the pipeline
    file and running peekingduck, you may refer to the
    :doc:`HelloCV tutorial </tutorials/01_hello_cv>`. |br|
    You should see a display of your camera with some instructions overlaid. Follow the instructions
    to position the checkerboard at 5 different positions in the camera. If the process is
    successful, the camera coefficients will be calculated and written to a file and you can start
    using the :mod:`augment.undistort` node.

    Inputs:
        |img_data|

    Outputs:
        |img_data|


    Configs:
        num_corners (:obj:`List[int]`):
            **default = [10, 7]**. |br|
            A list containing the number of internal corners along the vertical
            and horizontal axes. For example, in the given image above, the
            checkerboard is of size 11x8, so the number of internal corners is
            10x7. If you are using the given checkerboard above, you do not need
            to change this parameter.
        scale_factor (:obj:`int`):
            **default = 2**. |br|
            Factor to scale the image by when finding chessboard corners. For
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
        object_points_base = np.zeros((grid_height * grid_width, 3), np.float32)
        object_points_base[:, :2] = np_mgrid.T.reshape(-1, 2)

        # arrays to store object points and image points
        # points in real world
        self.object_points: List[np.ndarray] = [object_points_base] * NUM_PICTURES
        self.image_points: List[np.ndarray] = []  # points on image plane

        self.last_detection = time.time()
        self.num_detections = 0

        self.display_scales: Dict["str", Any["float", "int"]] = {}

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node calculates the camera distortion coefficients for undistortion.

        Args:
            inputs (dict): Inputs dictionary with the key `img`.

        Returns:
            outputs (dict): Outputs dictionary with the key `img`.
        """

        img = inputs["img"]
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img.shape[:2]

        self._initialize_display_scales(width)
        start_point, end_point, text_pos = _get_box_info(
            self.num_detections, width, height
        )

        detect_corners_success = False

        _draw_box(img, start_point, end_point, self.display_scales["box_thickness"])
        text_to_draw = [f"{DEFAULT_TEXT[0]} ({self.num_detections+1}/{NUM_PICTURES})"]

        # if sufficient time has passed, attempt to detect corners
        if time.time() - self.last_detection >= 5:
            detect_corners_success, corners = self._detect_corners(
                height, width, gray_img
            )

        # cv2 successfully detected the corners
        if detect_corners_success:
            corners_valid = _check_corners_validity(
                width, height, corners, start_point, end_point
            )

            if corners_valid == IMAGE_TOO_SMALL:
                text_to_draw = TOO_SMALL

        if detect_corners_success and corners_valid == CORNERS_OK:

            self.image_points.append(corners)
            self.num_detections += 1

            self._draw_text_and_corners(img, gray_img, corners, text_pos)

            # wait for user to press a key
            cv2.waitKey(0)
            self.last_detection = time.time()

            # if we have sufficient images, calculate the coefficients and write to a file
            if self.num_detections == NUM_PICTURES:
                calibration_data = self._calculate_coeffs(
                    img_shape=gray_img.shape[::-1]
                )
                self._write_coeffs(calibration_data)
                self._calculate_error(calibration_data)
                return {"pipeline_end": True}

        self._draw_text_and_countdown(img, text_to_draw, text_pos)

        return {"img": img}

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {"num_corners": List[int], "scale_factor": int, "file_path": str}

    def _initialize_display_scales(self, img_width: int) -> None:
        """Initializes display scales if it hasn't been initialized before"""
        if not self.display_scales:
            self.display_scales = {
                "box_thickness": max(int(img_width * BOX_THICKNESS_RATIO), 1),
                "normal_font_scale": img_width * NORMAL_FONT_SCALE_RATIO,
                "countdown_font_scale": img_width * COUNTDOWN_FONT_SCALE_RATIO,
                "normal_font_thickness": int(img_width * NORMAL_FONT_THICKNESS_RATIO),
                "countdown_font_thickness": int(
                    img_width * COUNTDOWN_FONT_THICKNESS_RATIO
                ),
            }

    def _detect_corners(self, height: int, width: int, gray_img: np.ndarray) -> tuple:
        """Detects corners in the image"""
        # downscale
        new_h = int(height / self.scale_factor)
        new_w = int(width / self.scale_factor)

        resized_img = cv2.resize(gray_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # try to find chessboard corners
        detect_corners_success, corners = cv2.findChessboardCorners(
            image=resized_img, patternSize=self.num_corners, corners=None
        )

        if corners is not None:
            corners = corners * self.scale_factor

        return detect_corners_success, corners

    def _calculate_coeffs(self, img_shape: tuple) -> tuple:
        """Performs calculations with detected corners"""
        calibration_data = cv2.calibrateCamera(
            objectPoints=self.object_points,
            imagePoints=self.image_points,
            imageSize=img_shape,
            cameraMatrix=None,
            distCoeffs=None,
        )

        (
            calibration_success,
            camera_matrix,
            distortion_coeffs,
            _,
            _,
        ) = calibration_data

        if calibration_success:
            self.logger.info("Calibration successful!")
            self.logger.info(f"Camera Matrix: {camera_matrix}")
            self.logger.info(f"Distortion Coefficients: {distortion_coeffs}")
        else:
            raise Exception("Calibration failed. Please try again.")

        return calibration_data

    def _write_coeffs(self, calibration_data: tuple) -> None:
        """Writes camera coefficients to a file"""
        (_, camera_matrix, distortion_coeffs, _, _) = calibration_data

        file_data = {}
        file_data["camera_matrix"] = camera_matrix.tolist()
        file_data["distortion_coeffs"] = distortion_coeffs.tolist()

        yaml.dump(file_data, open(self.file_path, "w"), default_flow_style=None)

    def _calculate_error(self, calibration_data: tuple) -> None:
        """Calculates re-projection error"""
        (
            _,
            camera_matrix,
            distortion_coeffs,
            rotation_vec,
            translation_vec,
        ) = calibration_data

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

        self.logger.info(f"Total error: {mean_error / len(self.object_points)}")

    def _draw_text_and_corners(
        self,
        img: np.ndarray,
        gray_img: np.ndarray,
        corners: np.ndarray,
        text_pos: tuple,
    ) -> None:
        """Draws text and corners on image"""
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
            image=img,
            patternSize=self.num_corners,
            corners=corners_accurate,
            patternWasFound=True,
        )

        if self.num_detections != NUM_PICTURES:
            text_to_draw = DETECTION_SUCCESS
        else:
            text_to_draw = DETECTION_COMPLETE

        _draw_text(
            img=img,
            texts=text_to_draw,
            pos_info=text_pos,
            font_scale=self.display_scales["normal_font_scale"],
            thickness=self.display_scales["normal_font_thickness"],
        )

        # display the image
        cv2.imshow("PeekingDuck", img)

    def _draw_text_and_countdown(
        self, img: np.ndarray, text_to_draw: List[str], text_pos: tuple
    ) -> None:
        """Draws text and countdown on image"""
        _draw_text(
            img=img,
            texts=text_to_draw,
            pos_info=text_pos,
            font_scale=self.display_scales["normal_font_scale"],
            thickness=self.display_scales["normal_font_thickness"],
        )

        time_to_next_detection = math.ceil(5 - time.time() + self.last_detection)
        if time_to_next_detection > 0:
            _draw_countdown(
                img=img,
                num=time_to_next_detection,
                font_scale=self.display_scales["countdown_font_scale"],
                thickness=self.display_scales["countdown_font_thickness"],
            )


def _get_box_info(num: int, width: int, height: int) -> tuple:
    """Returns start and end points of box, and position to put text"""
    start_points = {
        TOP_LEFT: (0, 0),
        TOP_RIGHT: (int(width * (1 - BOX_WIDTH_RATIO)), 0),
        BOTTOM_LEFT: (0, int(height * (1 - BOX_HEIGHT_RATIO))),
        BOTTOM_RIGHT: (
            int(width * (1 - BOX_WIDTH_RATIO)),
            int(height * (1 - BOX_HEIGHT_RATIO)),
        ),
        MIDDLE: (
            int(width * (1 / 2 - BOX_WIDTH_RATIO / 2)),
            int(height * (1 / 2 - BOX_HEIGHT_RATIO / 2)),
        ),
    }
    end_points = {
        TOP_LEFT: (int(width * BOX_WIDTH_RATIO), int(height * BOX_HEIGHT_RATIO)),
        TOP_RIGHT: (width, int(height * BOX_HEIGHT_RATIO)),
        BOTTOM_LEFT: (int(width * BOX_WIDTH_RATIO), height),
        BOTTOM_RIGHT: (width, height),
        MIDDLE: (
            int(width * (1 / 2 + BOX_WIDTH_RATIO / 2)),
            int(height * (1 / 2 + BOX_HEIGHT_RATIO / 2)),
        ),
    }
    text_positions = {
        TOP_LEFT: (TEXT_PADDING, int(height * BOX_HEIGHT_RATIO) - TEXT_PADDING),
        TOP_RIGHT: (
            int(width * (1 - BOX_WIDTH_RATIO)) + TEXT_PADDING,
            int(height * BOX_HEIGHT_RATIO) - TEXT_PADDING,
        ),
        BOTTOM_LEFT: (
            TEXT_PADDING,
            int(height * (1 - BOX_HEIGHT_RATIO)) + TEXT_PADDING,
        ),
        BOTTOM_RIGHT: (
            int(width * (1 - BOX_WIDTH_RATIO)) + TEXT_PADDING,
            int(height * (1 - BOX_HEIGHT_RATIO)) + TEXT_PADDING,
        ),
        MIDDLE: (
            int(width * (1 / 2 - BOX_WIDTH_RATIO / 2)) + TEXT_PADDING,
            int(height * (1 / 2 + BOX_HEIGHT_RATIO / 2)) - TEXT_PADDING,
        ),
    }
    pos_types = {
        TOP_LEFT: BOTTOM_LEFT,
        TOP_RIGHT: BOTTOM_LEFT,
        BOTTOM_LEFT: TOP_LEFT,
        BOTTOM_RIGHT: TOP_LEFT,
        MIDDLE: BOTTOM_LEFT,
    }

    return start_points[num], end_points[num], (text_positions[num], pos_types[num])


def _check_corners_validity(
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

    # if area is less than 1/4 of the box size
    if area < width * BOX_WIDTH_RATIO * height * BOX_HEIGHT_RATIO / 4:
        return IMAGE_TOO_SMALL

    # if the board is completely out of the box
    if (
        max_w < start_point[0]
        or end_point[0] < min_w
        or max_h < start_point[1]
        or end_point[1] < min_h
    ):
        return NOT_IN_BOX

    min_w_box = max(min_w, start_point[0])
    max_w_box = min(max_w, end_point[0])
    min_h_box = max(min_h, start_point[1])
    max_h_box = min(max_h, end_point[1])

    # check if at least 3 / 4 of the board area is within the box
    if (max_w_box - min_w_box) * (max_h_box - min_h_box) < area * AREA_THRESHOLD:
        return NOT_IN_BOX

    return CORNERS_OK


def _draw_box(
    img: np.ndarray, start_point: tuple, end_point: tuple, box_thickness: int
) -> None:
    """Draws rectangle on the image"""
    cv2.rectangle(
        img=img,
        pt1=start_point,
        pt2=end_point,
        color=(0, 0, 0),
        thickness=3 * box_thickness,
    )

    cv2.rectangle(
        img=img,
        pt1=start_point,
        pt2=end_point,
        color=CHAMPAGNE,
        thickness=box_thickness,
    )


def _draw_bgnd_box(
    img: np.ndarray,
    pt1: tuple,
    pt2: tuple,
) -> None:
    """Draws background box on image"""
    box_img = img.copy()

    # draw the rectangle
    cv2.rectangle(box_img, pt1, pt2, BLACK, cv2.FILLED)

    # apply the overlay
    cv2.addWeighted(box_img, BGND_BOX_OPACITY, img, 1 - BGND_BOX_OPACITY, 0, img)


def _draw_text(
    img: np.ndarray,
    texts: List[str],
    pos_info: tuple,
    font_scale: float,
    thickness: int,
) -> None:
    """Draws text on the image"""
    pos, pos_type = pos_info

    text_width = 0
    (_, text_height), baseline = cv2.getTextSize(
        text=texts[0],
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        thickness=thickness,
    )

    for text in texts:
        text_width = max(
            text_width,
            cv2.getTextSize(
                text=text,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                thickness=thickness,
            )[0][0],
        )

    img_width = img.shape[1]
    box_width = min(text_width + 10, int(img_width * BOX_WIDTH_RATIO - 10))
    if pos_type == TOP_LEFT:
        _draw_bgnd_box(
            img,
            (pos[0], pos[1]),
            (
                pos[0] + box_width,
                pos[1] + len(texts) * (text_height + baseline + 5) + baseline + 5,
            ),
        )

        pos = (pos[0] + 5, pos[1] + text_height + baseline + 5)

    elif pos_type == BOTTOM_LEFT:
        _draw_bgnd_box(
            img,
            (pos[0], pos[1] - len(texts) * (text_height + baseline + 5) - baseline - 5),
            (pos[0] + box_width, pos[1]),
        )

        pos = (
            pos[0] + 5,
            pos[1] - (text_height + baseline + 5) * (len(texts) - 1) - baseline - 5,
        )

    for text in texts:
        cv2.putText(
            img=img,
            text=text,
            org=pos,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=CHAMPAGNE,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        pos = (pos[0], pos[1] + text_height + baseline + 5)


def _draw_countdown(
    img: np.ndarray, num: int, font_scale: float, thickness: int
) -> None:
    """Draws a countdown in the center of the screen"""
    height, width = img.shape[:2]

    text = str(num)

    text_width, text_height = cv2.getTextSize(
        text=text,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        thickness=thickness,
    )[0]

    pos = (int(width / 2 - text_width / 2), int(height / 2 + text_height / 2))

    cv2.putText(
        img=img,
        text=text,
        org=pos,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=BLACK,
        thickness=thickness * 3,
        lineType=cv2.LINE_AA,
    )

    cv2.putText(
        img=img,
        text=text,
        org=pos,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=TOMATO,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
