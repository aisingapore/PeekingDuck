import numpy as np
import cv2


POSE_BBOX_COLOR = (255, 255, 0)
BLACK_COLOR = (0, 0, 0)
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
FPS_TEXT_COLOR = (255, 0, 255)
COUNTING_TEXT_COLOR = (0, 0, 255)
FONT_SCALE = 1
FONT_THICKNESS = 2
SKELETON_SHORT_NAMES = (
    "N", "LEY", "REY", "LEA", "REA", "LSH",
    "RSH", "LEL", "REL", "LWR", "RWR",
    "LHI", "RHI", "LKN", "RKN", "LAN", "RAN")


def draw_human_poses(image, poses, draw_activities=False):
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
        a, b = _project_points_onto_original_image(connection, image_size)
        cv2.line(frame, (a[0], a[1]), (b[0], b[1]), connection_color)

def _draw_keypoints(frame, keypoints, scores,
                    image_size, keypoint_dot_color):
    img_keypoints = _project_points_onto_original_image(
        keypoints, image_size)

    for idx, keypoint in enumerate(img_keypoints):
        _draw_one_keypoint_dot(frame, keypoint, keypoint_dot_color)
        if scores is not None:
            _draw_one_keypoint_text(frame, idx, scores, keypoint)

def _draw_one_keypoint_dot(frame, keypoint, keypoint_dot_color):
    cv2.circle(frame, (keypoint[0], keypoint[1]), 5, keypoint_dot_color, -1)

def _draw_one_keypoint_text(frame, idx, scores, keypoint):
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
    wt, ht = image_size[0], image_size[1]
    projected_points[:, 0] *= wt
    projected_points[:, 1] *= ht

    return projected_points

def draw_bboxes(frame, bboxes):
    '''draw only human bboxes onto frame image'''
    image_size = _get_image_size(frame)
    for bbox in bboxes:
        _draw_bbox(frame, bbox, image_size, HUMAN_BBOX_COLOR)

def draw_count(frame, count):
    """draw count of selected object onto frame

    Args:
        frame (List[float]): image of current frame
        count (int): total count of selected object
            in current frame
    """
    text = 'COUNT: {0}'.format(count)
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, COUNTING_TEXT_COLOR, 2, cv2.LINE_AA)
