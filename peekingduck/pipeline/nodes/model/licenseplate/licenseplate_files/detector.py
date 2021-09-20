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
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


class Detector:
    """Object detection class using yolo model to find object bboxes"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.root_dit = config["root"]
        self.class_labels = self._get_class_labels()
        self.yolo = self._create_yolo_model()

    def _get_class_labels(self) -> List[str]:
        classes_path = os.path.join(self.config["root"], self.config["classes"])
        with open(classes_path, "rt") as f:
            class_labels = f.read().rstrip("\n").split("\n")

        return class_labels

    def _create_yolo_model(self) -> cv.dnn_Net:
        """
        Creates yolo model for license plate detection
        """
        # model_type = self.config["model_type"]
        # self.model_configuration = os.path.join(
        #     self.config["root"], self.config["model_configuration"][model_type]
        # )
        # self.model_weights = os.path.join(
        #     self.config["root"], self.config["model_weights"][model_type]
        # )

        # # create Yolo model using opencv dnn API
        # model = cv.dnn.readNetFromDarknet(self.model_configuration, self.model_weights)
        # model.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        # model.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        model_type = self.config["model_type"]
        model_file = os.path.join(self.config['root'], self.config["model_weights_dir"][model_type])
        model = tf.saved_model.load(model_file, tags=[tag_constants.SERVING])

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

    def getOutputsNames(self) -> List[str]:
        """
        Returns the names of the output layers
        """
        # Get the names of all the layers in the network
        layersNames = self.yolo.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in self.yolo.getUnconnectedOutLayers()]

    def predict_object_bbox_from_image(
        self, image: np.array
    ) -> Tuple[List[np.array], List[str], List[float]]:
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

        # # using opencv for inference
        # frameHeight = image.shape[0]
        # frameWidth = image.shape[1]

        # # Create a 4D blob from a frame.
        # blob = cv.dnn.blobFromImage(
        #     image,
        #     1 / 255,
        #     (self.config["size"], self.config["size"]),
        #     [0, 0, 0],
        #     1,
        #     crop=False,
        # )

        # # Sets the input to the network
        # self.yolo.setInput(blob)

        # # Runs the forward pass to get output of the output layers
        # outputs = self.yolo.forward(self.getOutputsNames())

        # classIds = []
        # confidences = []
        # boxes = []

        # for out in outputs:
        #     for detection in out:
        #         # detections is of length 4 + 1 + num class
        #         # 4x bounding box , 1x box confidence, num of class
        #         # 5th element of detection onwards is the class confidence
        #         scores = detection[5:]
        #         classId = np.argmax(scores)
        #         # if scores[classId]>confThreshold:
        #         confidence = scores[classId]
        #         if confidence > self.config["confThreshold"]:
        #             center_x = int(detection[0] * frameWidth)
        #             center_y = int(detection[1] * frameHeight)
        #             width = int(detection[2] * frameWidth)
        #             height = int(detection[3] * frameHeight)
        #             left = int(center_x - width / 2)
        #             top = int(center_y - height / 2)
        #             classIds.append(classId)
        #             confidences.append(float(confidence))
        #             boxes.append([left, top, width, height])

        # # Run NMS and return the indices of bboxes to keep
        # indices = cv.dnn.NMSBoxes(
        #     boxes,
        #     confidences,
        #     self.config["confThreshold"],
        #     self.config["nmsThreshold"],
        # )
        # bboxes = []
        # scores = []
        # labels = []
        # for i in indices:
        #     i = i[0]
        #     box = boxes[i]
        #     left = box[0]
        #     top = box[1]
        #     width = box[2]
        #     height = box[3]
        #     right = left + width
        #     bot = top + height
        #     score = confidences[i]
        #     label = self.class_labels[classIds[i]]
        #     # label = "license_plate"
        #     bboxes.append(
        #         np.array(
        #             [
        #                 left / frameWidth,
        #                 top / frameHeight,
        #                 right / frameWidth,
        #                 bot / frameHeight,
        #             ]
        #         )
        #     )
        #     scores.append(score)
        #     labels.append(label)



        # # using opencv and tensorflow nms
        # for out in outputs:
        #     for detection in out:
        #         # detections is of length 4 + 1 + num class
        #         # bbox in (centerx,centery, w, h)
        #         # 4x bounding box , 1x box confidence, num of class
        #         # 5th element of detection onwards is the class confidence
        #         scores = detection[5:]
        #         num_classes = len(detection) - 5
        #         classId = np.argmax(scores)
        #         # if scores[classId]>confThreshold:
        #         confidence = scores[classId]
        #         if confidence > self.config["confThreshold"]:
        #             center_x = (detection[0])
        #             center_y = (detection[1])
        #             width = (detection[2])
        #             height = (detection[3])
        #             x_1 = (center_x - width / 2)
        #             y_1 = (center_y - height / 2)
        #             x_2 = (center_x + width / 2)
        #             y_2 = (center_y + height / 2)
        #             classIds.append(classId)
        #             confidences.append(float(confidence))
        #             boxes.append([y_1, x_1, y_2, x_2])

        # boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
        # confidences = tf.convert_to_tensor(confidences, dtype=tf.float32)

        # bboxes, scores, classes, nums = tf.image.combined_non_max_suppression(
        #     boxes=tf.reshape(boxes, (1, boxes.shape[0], 1, 4)),
        #     scores=tf.reshape(
        #         confidences, (1, confidences.shape[0], 1)),
        #     max_output_size_per_class=50,
        #     max_total_size=50,
        #     iou_threshold=0.5,
        #     score_threshold=0.1
        # )

        # labels = classes.numpy()[0]
        # labels = labels[:nums[0]]
        # bboxes = bboxes.numpy()[0]
        # bboxes = bboxes[:nums[0]]
        # scores = scores.numpy()[0]
        # scores = scores[:nums[0]]

        # bboxes[:, [0, 1]] = bboxes[:, [1, 0]]  # swapping x and y axes
        # bboxes[:, [2, 3]] = bboxes[:, [3, 2]]


        # using a converted weights file to tnesorflow saved model
        image_data = cv.resize(image , (self.config["size"],self.config["size"]))
        image_data = image_data/ 255.

        image_data = np.asarray([image_data]).astype(np.float32)
        infer = self.yolo.signatures['serving_default']
        pred_bbox = infer(tf.constant(image_data))
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        bboxes, scores, classes, nums = tf.image.combined_non_max_suppression(
                    boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                    scores=tf.reshape(
                        pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                    max_output_size_per_class=50,
                    max_total_size=50,
                    iou_threshold=0.3,
                    score_threshold=0.1
                )
        classes = classes.numpy()[0]
        classes = classes[:nums[0]]
        bboxes = bboxes.numpy()[0]
        bboxes = bboxes[:nums[0]]
        scores = scores.numpy()[0]
        scores = scores[:nums[0]]

        

        bboxes[:, [0, 1]] = bboxes[:, [1, 0]]  # swapping x and y axes
        bboxes[:, [2, 3]] = bboxes[:, [3, 2]]

        labels = [self.class_labels[int(i)] for i in classes]

        return bboxes, labels, scores
