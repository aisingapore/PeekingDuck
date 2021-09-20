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
        self.model_type = self.config["model_type"]

        model_file = os.path.join(
            self.config["root"], self.config["model_weights_dir"][self.model_type]
        )
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

    @staticmethod
    def bbox_scaling(bboxes, scale_factor):
        """
        To scale the width and height of bboxes from v4tiny
        After conversion of model from .cfg .weights from ,Alexey's Darknet repo,
        to tf model, bboxes are bigger. So downscaling required for better fit
        """
        for idx, box in enumerate(bboxes):
            x_1, y_1, x_2, y_2 = tuple(box)
            center_x = (x_1 + x_2) / 2
            center_y = (y_1 + y_2) / 2
            scaled_x_1 = center_x - ((x_2 - x_1) / 2 * scale_factor)
            scaled_x_2 = center_x + ((x_2 - x_1) / 2 * scale_factor)
            scaled_y_1 = center_y - ((y_2 - y_1) / 2 * scale_factor)
            scaled_y_2 = center_y + ((y_2 - y_1) / 2 * scale_factor)
            bboxes[idx] = [scaled_x_1, scaled_y_1, scaled_x_2, scaled_y_2]

        return bboxes

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
        # Use TF2 .pb saved model format for inference
        image_data = cv.resize(image, (self.config["size"], self.config["size"]))
        image_data = image_data / 255.0

        image_data = np.asarray([image_data]).astype(np.float32)
        infer = self.yolo.signatures["serving_default"]
        pred_bbox = infer(tf.constant(image_data))
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        # Use NMS to remove duplicate bboxes
        bboxes, scores, classes, nums = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])
            ),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.3,
            score_threshold=0.1,
        )
        classes = classes.numpy()[0]
        classes = classes[: nums[0]]
        bboxes = bboxes.numpy()[0]
        bboxes = bboxes[: nums[0]]
        scores = scores.numpy()[0]
        scores = scores[: nums[0]]

        bboxes[:, [0, 1]] = bboxes[:, [1, 0]]  # swapping x and y axes
        bboxes[:, [2, 3]] = bboxes[:, [3, 2]]

        # scaling of bboxes if v4tiny model is used
        if self.model_type == "v4tiny":
            bboxes = self.bbox_scaling(bboxes, 0.75)

        # update the labels names of the object detected
        labels = [self.class_labels[int(i)] for i in classes]

        return bboxes, labels, scores

