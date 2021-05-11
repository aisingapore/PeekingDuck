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

from typing import List, Dict, Tuple, Any
import numpy as np
import cv2


def draw_outputs(img: np.array, outputs: Tuple[List[np.array], List[float], List[str]],
                 class_names: Dict[str, Any]) -> np.array:
    """Draw object bounding box, confident score, and class name on
    the object in the image.

    Input:
        - img:      the image
        - outputs:  the outputs of prediction, which contain the
                    object' bounding box, confident score, and class
                    index
        - class_names: dictionary of classes string names

    Output:
        - img:      the image with the draw of outputs
    """
    boxes, objectness, classes = outputs
    width_height = np.flip(img.shape[0:2])
    for i, oneclass in enumerate(classes):
        x1y1 = tuple((np.array(boxes[i][0:2]) * width_height).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * width_height).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(class_names[oneclass], objectness[i]),
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img
