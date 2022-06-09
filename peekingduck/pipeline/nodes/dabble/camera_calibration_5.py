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

from typing import Any, Dict

import cv2, time, yaml
import numpy as np
from pathlib import Path

from peekingduck.pipeline.nodes.abstract_node import AbstractNode

# global constants
# terminal criteria for subpixel finetuning 
TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
TOP_LEFT = 0
TOP_RIGHT = 1
BOTTOM_LEFT = 2
BOTTOM_RIGHT = 3
MIDDLE = 4
INITIAL_INSTRUCTION = "PLACE THE BOARD IN THE BOX AS LARGE AS POSSIBLE"
TOO_SMALL = "THE BOARD IS TOO SMALL!"
OUT_OF_BOX = "MOVE THE BOARD INTO THE BOX"
DETECTION_SUCCESS = "DETECTION SUCCESSFUL! PRESS ANY KEY TO CONTINUE."
DETECTION_COMPLETE = "DETECTION COMPLETE! PRESS ANY KEY TO EXIT."

def get_optimal_font_scale(text: str, width):
    for scale in range(100, 0, -1):
        if scale / 50 >= 0.5:
            thickness = 2
        else:
            thickness = 1
        text_size = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale/50, thickness = thickness)
        new_width = text_size[0][0]
        if (new_width <= width):
            return scale / 50
    return 0.1

def draw_text(img: np.ndarray, text: str) -> None:
    """ Helper function to draw text on the image """

    img_copy = img.copy()

    height, width = img_copy.shape[:2]

    font_scale = get_optimal_font_scale(text, width - 10)

    text_size = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=2)
    text_width = text_size[0][0]
    text_height = text_size[0][1]

    # background box
    box_img = img_copy.copy()
    cv2.rectangle(
        box_img,
        (int(width / 2 - text_width / 2) - 5, int(height / 2 - text_height / 2) - 5),
        (int(width / 2 + text_width / 2) + 5, int(height / 2 + text_height / 2) + 5),
        (0, 0, 0),
        cv2.FILLED,
    )
    # apply the overlay
    cv2.addWeighted(box_img, 0.75, img_copy, 0.25, 0, img_copy)

    pos = (int(width / 2 - text_width / 2), int(height / 2 + text_height / 2))

    return cv2.putText(
        img = img_copy, 
        text = text,
        org = pos,
        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = font_scale,
        color = (0, 0, 255), #red
        thickness = 1,
        lineType = cv2.LINE_AA
    )

def draw_box(img: np.ndarray, start_point: tuple, end_point: tuple) -> None:
    """ Helper function to draw rectangle on the image """

    img_copy = img.copy()

    return cv2.rectangle(
        img = img_copy, 
        pt1 = start_point, 
        pt2 = end_point, 
        color = (0, 0, 255), #red
        thickness = 2
    )


class Node(AbstractNode):
    """Calculates camera coefficients for undistortion
    <https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html>'_.

    Inputs:
        |img_data|

    Outputs:
        num_detections (:obj:'int'):
            Count of number of successful detections so far.
        
        next_detection_in (:obj:'int'):
            Seconds remaining before the next detection.


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

        self.file_path = Path(self.file_path) # type: ignore
        # check if file_path has a '.yml' extension
        if self.file_path.suffix != '.yml':
            raise ValueError("Filepath must have a '.yml' extension.")
        if not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)

        self.grid_height = self.num_corners[0]
        self.grid_width = self.num_corners[1]

        # prepare all object points, like (0, 0, 0), (1, 0, 0), etc.
        self.object_points_base = np.zeros((self.grid_height * self.grid_width, 3), np.float32)
        self.object_points_base[:, :2] = np.mgrid[0:self.grid_height, 0:self.grid_width].T.reshape(-1, 2)

        # arrays to store object points and image points
        self.object_points = [] # points in real world
        self.image_points = [] # points on image plane

        self.last_detection = time.time()
        self.num_detections = 0

        self.num_pictures = 5


    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node calculates the camera distortion coefficients for undistortion.

        Args:
            inputs (dict): Inputs dictionary with the key `img`.

        Returns:
            outputs (dict): Outputs dictionary with the keys `img`, `num_detections`, `next_detection in`.
        """

        img = inputs["img"]
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detect_corners_success = False

        h, w = img.shape[:2]
        
        if self.num_detections == TOP_LEFT:
            start_point = (0, 0)
            end_point = (int(w/3), int(h/2))
            text_pos = (15, int(h/2) - 20)
        elif self.num_detections == TOP_RIGHT:
            start_point = (int(2*w/3), 0)
            end_point = (w, int(h/2))
            text_pos = (int(2*w/3) + 15, int(h/2) - 20)
        elif self.num_detections == BOTTOM_LEFT:
            start_point = (0, int(h/2))
            end_point = (int(w/3), h)
            text_pos = (15, int(h/2) + 20)
        elif self.num_detections == BOTTOM_RIGHT:
            start_point = (int(2*w/3), int(h/2))
            end_point = (w, h)
            text_pos = (int(2*w/3) + 15, int(h/2) + 20)
        else: #MIDDLE
            start_point = (int(w/3), int(h/4))
            end_point = (int(2*w/3), int(3*h/4))
            text_pos = (int(w/3) + 15, int(3*h/4) - 20)

        img_notext = draw_box(img, start_point, end_point)
        img = draw_text(img_notext, INITIAL_INSTRUCTION)

        # if sufficient time has passed
        if time.time() - self.last_detection >= 5:
            # downscale
            h, w = img.shape[:2]
            new_h = int(h / self.scale_factor)
            new_w = int(w / self.scale_factor)

            resized_img = cv2.resize(gray_img, (new_w, new_h), interpolation = cv2.INTER_AREA)

            # try to find chessboard corners
            detect_corners_success, corners = cv2.findChessboardCorners(resized_img, self.num_corners, None)

            if corners is not None:
                corners = corners * self.scale_factor

        # cv2 successfully detected the corners
        if detect_corners_success:
            
            h, w = img.shape[:2]
            min_w = w
            min_h = h
            max_w = 0
            max_h = 0
            for corner in corners:
                min_w = min(min_w, corner[0][0])
                max_w = max(max_w, corner[0][0])
                min_h = min(min_h, corner[0][1])
                max_h = max(max_h, corner[0][1])

            area = (max_w - min_w) * (max_h - min_h)

            if area < w * h / (6 * 4):
                img = draw_text(img_notext, TOO_SMALL)
                detect_corners_success = False

            else:
                min_w_box = max(min_w, start_point[0])
                max_w_box = min(max_w, end_point[0])
                min_h_box = max(min_h, start_point[1])
                max_h_box = min(max_h, end_point[1])

                area_in_box = max(0, (max_w_box - min_w_box) * (max_h_box - min_h_box))

                if area_in_box < 3 * area / 4:
                    img = draw_text(img_notext, OUT_OF_BOX)
                    detect_corners_success = False

        if detect_corners_success:

            self.object_points.append(self.object_points_base)
            self.image_points.append(corners)

            # improve corner accuracy
            corners_accurate = cv2.cornerSubPix(gray_img, corners, (11,11), (-1,-1), TERMINATION_CRITERIA)

            # draw corners and message on the image
            cv2.drawChessboardCorners(img_notext, self.num_corners, corners_accurate, detect_corners_success)

            self.num_detections += 1

            if self.num_detections != self.num_pictures:
                img = draw_text(img_notext, DETECTION_SUCCESS)
            else:
                img = draw_text(img_notext, DETECTION_COMPLETE)
            
            # display the image and wait for user to press a key
            cv2.imshow("PeekingDuck", img)
            cv2.waitKey(0)

            self.last_detection = time.time()

            # if we have sufficient images, calculate the coefficients and write to a file
            if self.num_detections == self.num_pictures:
                calibration_success, camera_matrix, distortion_coeffs, rotation_vec, translation_vec = cv2.calibrateCamera(self.object_points, self.image_points, gray_img.shape[::-1], None, None)

                if calibration_success:
                    self.logger.info(f"camera_matrix: {camera_matrix}")
                    self.logger.info(f"distortion_coeffs: {distortion_coeffs}")

                    file_data = {}
                    file_data['camera_matrix'] = camera_matrix.tolist()
                    file_data['distortion_coeffs'] = distortion_coeffs.tolist()

                    yaml.dump(file_data, open(self.file_path, "w"), default_flow_style = None)

                    self.logger.info(f"Calibration successful!")
                else:
                    raise Exception("Calibration failed. Please try again.")

                mean_error = 0
                for i in range(len(self.object_points)):
                    projected_image_points, _ = cv2.projectPoints(self.object_points[i], rotation_vec[i], translation_vec[i], camera_matrix, distortion_coeffs)
                    error = cv2.norm(self.image_points[i], projected_image_points, cv2.NORM_L2) / len(projected_image_points)
                    mean_error += error
                self.logger.info(f"total eror: {mean_error / len(self.object_points)}")

                return {"pipeline_end": True}

        # format text for the legend
        num_detections_string = f"{self.num_detections}/{self.num_pictures}"

        time_to_next_detection = max(0, int(5 - time.time() + self.last_detection))
        next_detection_in_string = f"{time_to_next_detection}s"

        return {"img": img, "num_detections": num_detections_string, "next_detection_in": next_detection_in_string}
