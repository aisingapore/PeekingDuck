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
import cv2
from cv2 import FONT_HERSHEY_SIMPLEX, LINE_AA


POSE_BBOX_COLOR = (255, 255, 0)
BLACK_COLOR = (0, 0, 0)
PINK_COLOR = (255, 0, 255)
ACTIVITY_COLOR = (100, 0, 255)
HUMAN_BBOX_COLOR = (255, 0, 255)
GROUP_BBOX_COLOR = (0, 140, 255)
HAND_BBOX_COLOR = (255, 0, 0)
OBJ_MASK_COLOR = (0, 100, 255)
KEYPOINT_TEXT_COLOR = (255, 0, 255)
KEYPOINT_DOT_COLOR = (0, 255, 0)
KEYPOINT_CONNECT_COLOR = (0, 255, 255)
HAND_KEYPOINT_DOT_COLOR = (0, 255, 0)
HAND_KEYPOINT_CONNECT_COLOR = (0, 0, 255)
COUNTING_TEXT_COLOR = (0, 0, 255)
FONT_SCALE = 1
FONT_THICKNESS = 2
SKELETON_SHORT_NAMES = (
    "N", "LEY", "REY", "LEA", "REA", "LSH",
    "RSH", "LEL", "REL", "LWR", "RWR",
    "LHI", "RHI", "LKN", "RKN", "LAN", "RAN")

def draw_human_poses(image, poses):
    '''draw pose estimates onto frame image'''
    image_size = _get_image_size(image)
    for pose in poses:
        if pose.bbox.shape == (2, 2):
            _draw_connections(image, pose.connections,
                              image_size, KEYPOINT_CONNECT_COLOR)
            _draw_keypoints(image, pose.keypoints,
                            pose.keypoint_scores, image_size,
                            KEYPOINT_DOT_COLOR)

def _get_image_size(frame):
    image_size = (frame.shape[1], frame.shape[0])  # width, height
    return image_size

def _draw_bbox(frame, bbox, image_size, color):
    top_left, bottom_right = _project_points_onto_original_image(bbox, image_size)
    cv2.rectangle(frame, (top_left[0], top_left[1]),
                  (bottom_right[0], bottom_right[1]),
                  color, 2)
    return top_left

def _draw_connections(frame, connections, image_size, connection_color):
    for connection in connections:
        pt1, pt2 = _project_points_onto_original_image(connection, image_size)
        cv2.line(frame, (pt1[0], pt1[1]), (pt2[0], pt2[1]), connection_color)

def _draw_keypoints(frame, keypoints, scores,
                    image_size, keypoint_dot_color):
    img_keypoints = _project_points_onto_original_image(
        keypoints, image_size)

    for idx, keypoint in enumerate(img_keypoints):
        _draw_one_keypoint_dot(frame, keypoint, keypoint_dot_color)
        if scores is not None:
            _draw_one_keypoint_text(frame, idx, keypoint)

def _draw_one_keypoint_dot(frame, keypoint, keypoint_dot_color):
    cv2.circle(frame, (keypoint[0], keypoint[1]), 5, keypoint_dot_color, -1)

def _draw_one_keypoint_text(frame, idx, keypoint):
    position = (keypoint[0], keypoint[1])
    text = str(SKELETON_SHORT_NAMES[idx])

    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                0.4, KEYPOINT_TEXT_COLOR, 1, cv2.LINE_AA)

def _project_points_onto_original_image(points, image_size):
    """Project points from relative value to absolute values in original
    image.  E.g. from (1, 0.5) to (1280, 400).  It use a coordinate with
    original point (0, 0) at top-left.

    args:
        - points: np.array of (x, y) pairs of normalized joint coordinates.
                    i.e X and Y are in the range [0.0, 1.0]
        - image_size: image shape tuple to project (width, height)

    return:
        list of (x, y) pairs of joint coordinates transformed to image
        coordinates. x will be in the range [0, image width]. y will be in
        in the range [0, image height]
    """
    if len(points) == 0:
        return []

    points = points.reshape((-1, 2))

    projected_points = np.array(points, dtype=np.float32)
    width, height = image_size[0], image_size[1]
    projected_points[:, 0] *= width
    projected_points[:, 1] *= height

    return projected_points

def draw_human_bboxes(frame, human_bboxes):
    '''draw only human bboxes onto frame image'''
    image_size = _get_image_size(frame)
    for bbox in human_bboxes:
        _draw_bbox(frame, bbox, image_size, HUMAN_BBOX_COLOR)

def _draw_bbox(frame, bbox, image_size, color):
    top_left, bottom_right = _project_points_onto_original_image(bbox, image_size)
    cv2.rectangle(frame, (top_left[0], top_left[1]),
                  (bottom_right[0], bottom_right[1]),
                  color, 2)
    return top_left

def draw_fps(frame: np.array, current_fps: float) -> None:
    """ Draw FPS onto frame image

    args:
        - frame: array containing the RGB values of the frame image
        - current_fps: value of the calculated FPS
    """
    text = "FPS: {:.05}".format(current_fps)
    text_location = (25, 25)

    cv2.putText(frame, text, text_location, FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                PINK_COLOR, FONT_THICKNESS, LINE_AA)
