# Copyright 2021 AI Singapore

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Object detection class using yolo single label model 
to find license plate object bboxes
"""

import cv2 as cv
import logging
import os as os
from typing import Dict, Any, List, Tuple
import numpy as np


class Detector:
    """Object detection class using yolo model to find object bboxes"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.root_dit = config["root"]
        self.class_labels = self._get_class_labels()
        self.yolo = self._create_yolo_model()

    def _get_class_labels(self):
        classes_path = os.path.join(self.config["root"], self.config["classes"])
        with open(classes_path, "rt") as f:
            class_labels = f.read().rstrip("\n").split("\n")

        return class_labels

    def _create_yolo_model(self):
        """
        Creates yolo model for license plate detection
        """
        model_type = self.config["model_type"]
        self.model_configuration = os.path.join(
            self.config["root"], self.config["model_configuration"][model_type]
        )
        self.model_weights = os.path.join(
            self.config["root"], self.config["model_weights"][model_type]
        )

        # create Yolo model using opencv dnn API
        model = cv.dnn.readNetFromDarknet(self.model_configuration, self.model_weights)
        model.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        model.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        self.logger.info(
            "Yolo model loaded with following configs: \n \
            Model type: %s, \n \
            Input resolution: %s, \n \
            NMS threshold: %s, \n \
            Score threshold: %s",
            self.config["model_type"],
            self.config["size"],
            self.config["nmsThreshold"],
            self.config["confThreshold"],
        )

        return model

    def getOutputsNames(self):
        """
        Returns the names of the output layers
        """
        # Get the names of all the layers in the network
        layersNames = self.yolo.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in self.yolo.getUnconnectedOutLayers()]

    def predict_object_bbox_from_image(self, image: np.array):
        """Detect all objects' bounding box from one image

        args:
            - yolo:  (Model) model like yolov4 or yolov4_tiny
            - image: (np.array) input image

        return:
            - boxes: (np.array) an array of bounding box with
                    definition like (x1, y1, x2, y2), in a
                    coordinate system with original point in
                    the left top corner
        """
        frameHeight = image.shape[0]
        frameWidth = image.shape[1]

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(
            image,
            1 / 255,
            (self.config["size"], self.config["size"]),
            [0, 0, 0],
            1,
            crop=False,
        )

        # Sets the input to the network
        self.yolo.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outputs = self.yolo.forward(self.getOutputsNames())

        classIds = []
        confidences = []
        boxes = []

        for out in outputs:
            for detection in out:

                scores = detection[5:]
                classId = np.argmax(scores)
                # if scores[classId]>confThreshold:
                confidence = scores[classId]
                if confidence > self.config["confThreshold"]:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Run NMS and return the indices of bboxes to keep
        indices = cv.dnn.NMSBoxes(
            boxes,
            confidences,
            self.config["confThreshold"],
            self.config["nmsThreshold"],
        )
        bboxes = []
        scores = []
        labels = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            right = left + width
            bot = top + height
            score = confidences[i]
            label = self.class_labels[classIds[i]]
            bboxes.append(
                np.array(
                    [
                        left / frameWidth,
                        top / frameHeight,
                        right / frameWidth,
                        bot / frameHeight,
                    ]
                )
            )
            scores.append(score)
            labels.append(label)

        return bboxes, labels, scores
