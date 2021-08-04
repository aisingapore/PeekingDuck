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
functions for drawing bounding box related UI components
"""

from typing import List, Tuple
import numpy as np
import cv2
from cv2 import FONT_HERSHEY_SIMPLEX, LINE_AA
from peekingduck.pipeline.nodes.draw.utils.constants import \
    CHAMPAGNE, BLACK, THICK, VERY_THICK, NORMAL_FONTSCALE, \
    POINT_RADIUS, FILLED, PRIMARY_PALETTE, PRIMARY_PALETTE_LENGTH as TOTAL_COLOURS
from peekingduck.pipeline.nodes.draw.utils.general import \
    get_image_size, project_points_onto_original_image


def draw_bboxes(frame: np.array,
                bboxes: List[List[float]],
                bbox_labels: List[str],
                show_labels: bool) -> None:
    """Draw bboxes onto an image frame.

    Args:
        frame (np.array): image of current frame
        bboxes (List[List[float]]): bounding box coordinates
        colour (Tuple[int, int, int]): colour used for bounding box
        bbox_labels (List[str]): labels of object detected
    """
    image_size = get_image_size(frame)
    # Get unique label colour indexes
    colour_indx = {label: indx for indx, label in enumerate(set(bbox_labels))}

    for i, bbox in enumerate(bboxes):
        colour = PRIMARY_PALETTE[colour_indx[bbox_labels[i]] % TOTAL_COLOURS]
        if show_labels:
            _draw_bbox(frame, bbox, image_size,
                       colour, bbox_labels[i])
        else:
            _draw_bbox(frame, bbox, image_size, colour)

def _draw_bbox(frame: np.array,
               bbox: List[float],
               image_size: Tuple[int, int],
               colour: Tuple[int, int, int],
               bbox_label: str = None) -> np.array:
    """ Draw a single bounding box """
    top_left, bottom_right = project_points_onto_original_image(
        bbox, image_size)
    cv2.rectangle(frame, (top_left[0], top_left[1]),
                  (bottom_right[0], bottom_right[1]),
                  colour, VERY_THICK)

    if bbox_label:
        _draw_label(frame, top_left, bbox_label, colour, BLACK)


def _draw_label(frame: np.array,
                top_left: Tuple[int, int],
                bbox_label: str,
                bg_colour: Tuple[int, int, int],
                text_colour: Tuple[int, int, int]) -> None:
    """Draw bbox label at top left of bbox"""
        # get label size
    (text_width, text_height), baseline = cv2.getTextSize(bbox_label,
                                                          FONT_HERSHEY_SIMPLEX,
                                                          NORMAL_FONTSCALE,
                                                          THICK)
    # put filled text rectangle
    cv2.rectangle(frame,
                  (top_left[0], top_left[1]),
                  (int(top_left[0] + text_width),
                   int(top_left[1] - text_height - baseline)),
                  bg_colour,
                  FILLED)

    # put text above rectangle
    bbox_label = bbox_label[:1].capitalize() + bbox_label[1:]
    cv2.putText(frame, bbox_label, (top_left[0], int(top_left[1]-6)), FONT_HERSHEY_SIMPLEX,
                NORMAL_FONTSCALE, text_colour, THICK, LINE_AA)


def draw_tags(frame: np.array,
              bboxes: List[List[float]],
              tags: List[str],
              colour: Tuple[int, int, int]) -> None:
    """Draw tags above bboxes.

    Args:
        frame (np.array): image of current frame
        bboxes (List[List[float]]): bounding box coordinates
        tags (List[string]): tag associated with bounding box
        color (Tuple[int, int, int]): color of text
    """
    image_size = get_image_size(frame)
    for idx, bbox in enumerate(bboxes):
        _draw_tag(frame, bbox, tags[idx], image_size, colour)


def _draw_tag(frame: np.array,
              bbox: np.array,
              tag: str,
              image_size: Tuple[int, int],
              colour: Tuple[int, int, int]) -> None:
    """Draw a tag above a single bounding box.
    """
    top_left, btm_right = project_points_onto_original_image(bbox, image_size)

    # Find offset to centralize text
    (text_width, _), baseline = cv2.getTextSize(tag,
                                                FONT_HERSHEY_SIMPLEX,
                                                NORMAL_FONTSCALE,
                                                THICK)
    bbox_width = btm_right[0] - top_left[0]
    offset = int((bbox_width - text_width) / 2)
    position = (int(top_left[0] + offset), int(top_left[1] - baseline))
    cv2.putText(frame, tag, position, FONT_HERSHEY_SIMPLEX,
                NORMAL_FONTSCALE, colour, VERY_THICK)



def draw_pts(frame: np.array, pts: List[Tuple[float]]) -> None:
    """draw pts of selected object onto frame

    Args:
        frame (np.array): image of current frame
        pts (List[Tuple[float]]): bottom midpoints of bboxes
    """
    for point in pts:
        cv2.circle(frame, point, POINT_RADIUS, CHAMPAGNE, -1)
