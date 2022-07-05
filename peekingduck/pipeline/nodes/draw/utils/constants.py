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
Constants used for drawing
"""

# colors
CHAMPAGNE = (156, 223, 244)
BLIZZARD = (241, 232, 164)
VIOLET_BLUE = (188, 118, 119)
TOMATO = (77, 103, 255)
BROWN = (96, 109, 167)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
PRIMARY_PALETTE_LENGTH = 5
PRIMARY_PALETTE = [CHAMPAGNE, BLIZZARD, TOMATO, VIOLET_BLUE, BROWN]

# constants for thickness
THIN = 1
THICK = 2
VERY_THICK = 3

# constant to fill shapes in cv2. To be replace line thickness
FILLED = -1

# constants for font scale
SMALL_FONTSCALE = 0.5
NORMAL_FONTSCALE = 1
BIG_FONTSCALE = 2

# constants used for image manipulation
LOWER_SATURATION = 0.5

# constants for pts
POINT_RADIUS = 5

# constants for masks
SATURATION_STEPS = 8
SATURATION_MINIMUM = 100
ALPHA = 0.5
CONTOUR_COLOR = (0, 0, 0)
DEFAULT_CLASS_COLOR = (204, 204, 204)
# fmt: off
CLASS_COLORS = {
    "person": (23,127,245),

    "bicycle":      (146,27,49),
    "car":          (146,27,49),
    "motorcycle":   (146,27,49),
    "airplane":     (146,27,49),
    "bus":          (146,27,49),
    "train":        (146,27,49),
    "truck":        (146,27,49),
    "boat":         (146,27,49),

    "traffic light":    (79,14,136),
    "fire hydrant":     (79,14,136),
    "stop sign":        (79,14,136),
    "parking meter":    (79,14,136),
    "bench":            (79,14,136),

    "bird":     (100,96,0),
    "cat":      (100,96,0),
    "dog":      (100,96,0),
    "horse":    (100,96,0),
    "sheep":    (100,96,0),
    "cow":      (100,96,0),
    "elephant": (100,96,0),
    "bear":     (100,96,0),
    "zebra":    (100,96,0),
    "giraffe":  (100,96,0),

    "backpack": (0,51,0),
    "umbrella": (0,51,0),
    "handbag":  (0,51,0),
    'tie':      (0,51,0),
    "suitcase": (0,51,0),

    "frisbee":          (140,20,74),
    "skis":             (140,20,74),
    "snowboard":        (140,20,74),
    "sports ball":      (140,20,74),
    "kite":             (140,20,74),
    "baseball bat":     (140,20,74),
    "baseball glove":   (140,20,74),
    "skateboard":       (140,20,74),
    "surfboard":        (140,20,74),
    "tennis racket":    (140,20,74),

    "bottle":       (0,111,255),
    "wine glass":   (0,111,255),
    "cup":          (0,111,255),
    "fork":         (0,111,255),
    "knife":        (0,111,255),
    "spoon":        (0,111,255),
    "bowl":         (0,111,255),

    "banana":   (23,119,130),
    "apple":    (23,119,130),
    "sandwich": (23,119,130),
    "orange":   (23,119,130),
    "broccoli": (23,119,130),
    "carrot":   (23,119,130),
    "hot dog":  (23,119,130),
    "pizza":    (23,119,130),
    "donut":    (23,119,130),
    "cake":     (23,119,130),

    "chair":        (155,87,1),
    "couch":        (155,87,1),
    "potted plant": (155,87,1),
    "bed":          (155,87,1),
    "dining table": (155,87,1),
    "toilet":       (155,87,1),

    "tv":           (12,54,191),
    "laptop":       (12,54,191),
    "mouse":        (12,54,191),
    "remote":       (12,54,191),
    "keyboard":     (12,54,191),
    "cell phone":   (12,54,191),

    "microwave":    (0,111,255),
    "oven":         (0,111,255),
    "toaster":      (0,111,255),
    "sink":         (0,111,255),
    "refrigerator": (0,111,255),

    "book":         (35,39,62),
    "clock":        (35,39,62),
    "vase":         (35,39,62),
    "scissors":     (35,39,62),
    "teddy bear":   (35,39,62),
    "hair drier":   (35,39,62),
    "toothbrush":   (35,39,62),
}
# fmt: on
