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
    "person": (245,127,23),

    "bicycle":      (49,27,146),
    "car":          (49,27,146),
    "motorcycle":   (49,27,146),
    "airplane":     (49,27,146),
    "bus":          (49,27,146),
    "train":        (49,27,146),
    "truck":        (49,27,146),
    "boat":         (49,27,146),

    "traffic light":    (136,14,79),
    "fire hydrant":     (136,14,79),
    "stop sign":        (136,14,79),
    "parking meter":    (136,14,79),
    "bench":            (136,14,79),

    "bird":     (0,96,100),
    "cat":      (0,96,100),
    "dog":      (0,96,100),
    "horse":    (0,96,100),
    "sheep":    (0,96,100),
    "cow":      (0,96,100),
    "elephant": (0,96,100),
    "bear":     (0,96,100),
    "zebra":    (0,96,100),
    "giraffe":  (0,96,100),

    "backpack": (0,51,0),
    "umbrella": (0,51,0),
    "handbag":  (0,51,0),
    'tie':      (0,51,0),
    "suitcase": (0,51,0),

    "frisbee":          (74,20,140),
    "skis":             (74,20,140),
    "snowboard":        (74,20,140),
    "sports ball":      (74,20,140),
    "kite":             (74,20,140),
    "baseball bat":     (74,20,140),
    "baseball glove":   (74,20,140),
    "skateboard":       (74,20,140),
    "surfboard":        (74,20,140),
    "tennis racket":    (74,20,140),

    "bottle":       (255,111,0),
    "wine glass":   (255,111,0),
    "cup":          (255,111,0),
    "fork":         (255,111,0),
    "knife":        (255,111,0),
    "spoon":        (255,111,0),
    "bowl":         (255,111,0),

    "banana":   (130,119,23),
    "apple":    (130,119,23),
    "sandwich": (130,119,23),
    "orange":   (130,119,23),
    "broccoli": (130,119,23),
    "carrot":   (130,119,23),
    "hot dog":  (130,119,23),
    "pizza":    (130,119,23),
    "donut":    (130,119,23),
    "cake":     (130,119,23),

    "chair":        (1,87,155),
    "couch":        (1,87,155),
    "potted plant": (1,87,155),
    "bed":          (1,87,155),
    "dining table": (1,87,155),
    "toilet":       (1,87,155),

    "tv":           (191,54,12),
    "laptop":       (191,54,12),
    "mouse":        (191,54,12),
    "remote":       (191,54,12),
    "keyboard":     (191,54,12),
    "cell phone":   (191,54,12),

    "microwave":    (255,111,0),
    "oven":         (255,111,0),
    "toaster":      (255,111,0),
    "sink":         (255,111,0),
    "refrigerator": (255,111,0),

    "book":         (62,39,35),
    "clock":        (62,39,35),
    "vase":         (62,39,35),
    "scissors":     (62,39,35),
    "teddy bear":   (62,39,35),
    "hair drier":   (62,39,35),
    "toothbrush":   (62,39,35),
}
# fmt: on
