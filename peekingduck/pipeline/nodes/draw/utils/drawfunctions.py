# Copyright 2021 AI Singapore
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
Utils, constants, and functions for drawing
"""

from typing import List, Tuple, Any, Iterable, Union
import numpy as np
import cv2
from cv2 import FONT_HERSHEY_SIMPLEX, LINE_AA

POSE_BBOX_COLOR = (255, 255, 0)
BLACK_COLOR = (0, 0, 0)
PINK_COLOR = (255, 0, 255)
ACTIVITY_COLOR = (100, 0, 255)
OBJ_MASK_COLOR = (0, 100, 255)
KEYPOINT_DOT_COLOR = (0, 255, 0)
COUNTING_TEXT_COLOR = (0, 0, 255)
COLOUR_SET_LENGTH = 10
COLOUR_SET = [(255, 0, 0),
              (0, 255, 0),
              (0, 0, 255),
              (255, 255, 0),
              (0, 255, 255),
              (255, 0, 255),
              (255, 165, 0),
              (128, 0, 128),
              (255, 20, 147),
              (255, 250, 250)]
FONT_SCALE = 0.5
FONT_THICKNESS = 1
SKELETON_SHORT_NAMES = (
    "N", "LEY", "REY", "LEA", "REA", "LSH",
    "RSH", "LEL", "REL", "LWR", "RWR",
    "LHI", "RHI", "LKN", "RKN", "LAN", "RAN")

SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4],
            [3, 5], [4, 6], [5, 7]]


def draw_human_poses(image: np.array,
                     keypoints: np.ndarray,
                     keypoint_scores: np.ndarray,
                     keypoint_conns: np.ndarray,
                     keypoint_dot_color: Tuple[int, int, int],
                     keypoint_dot_radius: int,
                     keypoint_connect_color: Tuple[int, int, int],
                     keypoint_text_color: Tuple[int, int, int]) -> None:
    # pylint: disable=too-many-arguments
    """Draw poses onto an image frame.

    Args:
        image (np.array): image of current frame
        keypoints (List[Any]): list of keypoint coordinates
        keypoints_scores (List[Any]): list of keypoint scores
        keypoints_conns (List[Any]): list of keypoint connections
        keypoint_dot_color (Tuple[int, int, int]): color of keypoint
        keypoint_dot_radius (int): radius of keypoint
        keypoint_connect_color (Tuple[int, int, int]): color of joint
        keypoint_text_color (Tuple[int, int, int]): color of keypoint names
    """
    image_size = _get_image_size(image)
    num_persons = keypoints.shape[0]
    if num_persons > 0:
        for i in range(num_persons):
            _draw_connections(image, keypoint_conns[i],
                              image_size, keypoint_connect_color)
            _draw_keypoints(image, keypoints[i],
                            keypoint_scores[i], image_size,
                            keypoint_dot_color, keypoint_dot_radius, keypoint_text_color)


def _get_image_size(frame: np.array) -> Tuple[int, int]:
    """ Obtain image size of input frame """
    image_size = (frame.shape[1], frame.shape[0])  # width, height
    return image_size


def _draw_connections(frame: np.array,
                      connections: Union[None, Iterable[Any]],
                      image_size: Tuple[int, int],
                      connection_color: Tuple[int, int, int]) -> None:
    """ Draw connections between detected keypoints """
    if connections is not None:
        for connection in connections:
            pt1, pt2 = _project_points_onto_original_image(connection, image_size)
            cv2.line(frame, (pt1[0], pt1[1]), (pt2[0], pt2[1]), connection_color)


def _draw_keypoints(frame: np.ndarray,
                    keypoints: np.ndarray,
                    scores: np.ndarray,
                    image_size: Tuple[int, int],
                    keypoint_dot_color: Tuple[int, int, int],
                    keypoint_dot_radius: int,
                    keypoint_text_color: Tuple[int, int, int]) -> None:
    # pylint: disable=too-many-arguments
    """ Draw detected keypoints """
    img_keypoints = _project_points_onto_original_image(
        keypoints, image_size)

    for idx, keypoint in enumerate(img_keypoints):
        _draw_one_keypoint_dot(frame, keypoint, keypoint_dot_color, keypoint_dot_radius)
        if scores is not None:
            _draw_one_keypoint_text(frame, idx, keypoint, keypoint_text_color)


def _draw_one_keypoint_dot(frame: np.ndarray,
                           keypoint: np.ndarray,
                           keypoint_dot_color: Tuple[int, int, int],
                           keypoint_dot_radius: int) -> None:
    """ Draw single keypoint """
    cv2.circle(frame, (keypoint[0], keypoint[1]), keypoint_dot_radius, keypoint_dot_color, -1)


def _draw_one_keypoint_text(frame: np.ndarray,
                            idx: int,
                            keypoint: np.ndarray,
                            keypoint_text_color: Tuple[int, int, int]) -> None:
    """ Draw name above keypoint """
    position = (keypoint[0], keypoint[1])
    text = str(SKELETON_SHORT_NAMES[idx])

    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                0.4, keypoint_text_color, 1, cv2.LINE_AA)


def _project_points_onto_original_image(points: np.ndarray,
                                        image_size: Tuple[int, int]) -> np.ndarray:
    """ Project points from relative value (0, 1) to absolute values in original
    image. Note that coordinate (0, 0) starts from image top-left. """
    if len(points) == 0:
        return []

    points = points.reshape((-1, 2))

    projected_points = np.array(points, dtype=np.float32)

    width, height = image_size[0], image_size[1]
    projected_points[:, 0] *= width
    projected_points[:, 1] *= height

    return projected_points


def draw_bboxes(frame: np.array,
                bboxes: List[List[float]],
                color: Tuple[int, int, int],
                thickness: int) -> None:
    """Draw bboxes onto an image frame.

    Args:
        frame (np.array): image of current frame
        bboxes (List[List[float]]): bounding box coordinates
        color (Tuple[int, int, int]): color of bounding box
        thickness (int): thickness of bounding box
    """
    image_size = _get_image_size(frame)
    for bbox in bboxes:
        _draw_bbox(frame, bbox, image_size, color, thickness)


def _draw_bbox(frame: np.array,
               bbox: List[float],
               image_size: Tuple[int, int],
               color: Tuple[int, int, int],
               thickness: int) -> np.array:
    """ Draw a single bounding box """
    top_left, bottom_right = _project_points_onto_original_image(
        bbox, image_size)
    cv2.rectangle(frame, (top_left[0], top_left[1]),
                  (bottom_right[0], bottom_right[1]),
                  color, thickness)


def draw_tags(frame: np.array,
              bboxes: List[List[float]],
              tags: List[str],
              color: Tuple[int, int, int]) -> None:
    """Draw tags above bboxes.

    Args:
        frame (np.array): image of current frame
        bboxes (List[List[float]]): bounding box coordinates
        tags (List[string]): tag associated with bounding box
        color (Tuple[int, int, int]): color of text
    """
    image_size = _get_image_size(frame)
    for idx, bbox in enumerate(bboxes):
        _draw_tag(frame, bbox, tags[idx], image_size, color)


def _draw_tag(frame: np.array,
              bbox: np.array,
              tag: str,
              image_size: Tuple[int, int],
              color: Tuple[int, int, int]) -> None:
    """Draw a tag above a single bounding box.
    """
    top_left, _ = _project_points_onto_original_image(bbox, image_size)
    position = int(top_left[0]), int(top_left[1]-25)
    cv2.putText(frame, tag, position, FONT_HERSHEY_SIMPLEX, 1, color, 2)


def draw_count(frame: np.array, count: int) -> None:
    """draw count of selected object onto frame

    Args:
        frame (np.array): image of current frame
        count (int): total count of selected object
            in current frame
    """
    text = 'COUNT: {0}'.format(count)
    cv2.putText(frame, text, (10, 50), FONT_HERSHEY_SIMPLEX,
                0.75, COUNTING_TEXT_COLOR, 2, LINE_AA)


def draw_pts(frame: np.array, pts: List[Tuple[float]]) -> None:
    """draw pts of selected object onto frame

    Args:
        frame (np.array): image of current frame
        pts (List[Tuple[float]]): bottom midpoints of bboxes
    """
    for point in pts:
        cv2.circle(frame, point, 5, KEYPOINT_DOT_COLOR, -1)


def draw_fps(frame: np.array, current_fps: float) -> None:
    """ Draw FPS onto frame image
<<<<<<< HEAD
    args:
        - frame: array containing the RGB values of the frame image
        - current_fps: value of the calculated FPS
=======

    Args:
        frame (np.array): image of current frame
        current_fps (float): value of the calculated FPS
>>>>>>> 823b177166ef1f40faab8e8611001c73a6b634c7
    """
    text = "FPS: {:.05}".format(current_fps)
    text_location = (25, 25)

    cv2.putText(frame, text, text_location, FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                PINK_COLOR, FONT_THICKNESS, LINE_AA)


def _draw_zone_area(frame: np.array, points: List[Tuple[int]],
                    zone_index: int) -> None:
    total_points = len(points)
    for i in range(total_points):
        if i == total_points-1:
            # for last point, link to first point
            cv2.line(frame, points[i], points[0],
                     COLOUR_SET[zone_index % COLOUR_SET_LENGTH], 3)
        else:
            # for all other points, link to next point in polygon
            cv2.line(frame, points[i], points[i+1],
                     COLOUR_SET[zone_index % COLOUR_SET_LENGTH], 3)


def draw_zones(frame: np.array, zones: List[Any]) -> None:
    """draw the boundaries of the zones used in zoning analytics

    Args:
        frame (np.array): image of current frame
        zones (Zone): zones used in the zoning analytics. possible
        classes are Area and Divider.
    """
    for i, zone_pts in enumerate(zones):
        _draw_zone_area(frame, zone_pts, i)


def draw_zone_count(frame: np.array, zone_count: List[int]) -> None:
    """draw pts of selected object onto frame

    Args:
        frame (np.array): image of current frame
        zone_count (List[float]): object count, likely people, of each zone used
        in the zone analytics
    """
    y_pos = 50
    text = '--ZONE COUNTS--'
    cv2.putText(frame, text, (25, y_pos), FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                COUNTING_TEXT_COLOR, 2, LINE_AA)
    for i, count in enumerate(zone_count):
        y_pos += 25
        text = 'ZONE {0}: {1}'.format(i+1, count)
        cv2.putText(frame, text, (25, y_pos), FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                    COLOUR_SET[i % COLOUR_SET_LENGTH], 2, LINE_AA)
