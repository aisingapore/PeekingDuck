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
Constants used for drawing
"""

# colours
CHAMPAGNE = (156, 223, 244)
BLIZZARD = (241, 232, 164)
VIOLET_BLUE = (188, 118, 119)
TOMATO = (77, 103, 255)
BROWN = (96,109, 167)
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
