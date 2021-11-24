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

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


class Detector:
    """Object detection class using yolo model to find object bboxes"""

    def __init__(self, config: Dict[str, Any], model_dir: Path) -> None:
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.model_dir = model_dir
        self.class_labels = self._get_class_labels()
        self.yolo = self._create_yolo_model()

    def _get_class_labels(self) -> List[str]:
        classes_path = self.model_dir / self.config["weights"]["classes_file"]
        with open(classes_path, "rt") as file:
            class_labels = file.read().rstrip("\n").split("\n")

        return class_labels

    def _create_yolo_model(self) -> cv.dnn_Net:
        """
        Creates yolo model for license plate detection
        """
        self.model_type = self.config["model_type"]
        model_path = (
            self.model_dir
            / self.config["weights"]["saved_model_subdir"][self.model_type]
        )
        model = tf.saved_model.load(str(model_path), tags=[tag_constants.SERVING])

        self.logger.info(
            "Yolo model loaded with following configs: \n\t"
            f"Model type: {self.config['model_type']}, \n\t"
            f"Input resolution: {self.config['size']}, \n\t"
            f"NMS threshold: {self.config['yolo_iou_threshold']}, \n\t"
            f"Score threshold: {self.config['yolo_score_threshold']}"
        )

        return model

    @staticmethod
    def bbox_scaling(bboxes: List[list], scale_factor: float) -> List[list]:
        """
        To scale the width and height of bboxes from v4tiny
        After the conversion of the model in .cfg and .weight file format, from
        Alexey's Darknet repo, to tf model, bboxes are bigger.
        So downscaling is required for a better fit
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
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect all license plate objects' bounding box from one image

        args:
                image: (Numpy Array) input image

        return:
                boxes: (Numpy Array) an array of bounding box with
                    definition like (x1, y1, x2, y2), in a
                    coordinate system with origin point in
                    the left top corner
                labels: (Numpy Array) an array of class labels
                scores: (Numpy Array) an array of confidence scores
        """
        # Use TF2 .pb saved model format for inference
        image_data = cv.resize(image, (self.config["size"], self.config["size"]))
        image_data = image_data / 255.0

        image_data = np.asarray([image_data]).astype(np.float32)
        infer = self.yolo.signatures["serving_default"]
        pred_bbox = infer(tf.constant(image_data))
        for _, value in pred_bbox.items():
            pred_conf = value[:, :, 4:]
            boxes = value[:, :, 0:4]

        # performs nms using model's predictions
        bboxes, scores, classes, nums = tf.image.combined_non_max_suppression(
            tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])
            ),
            self.config["max_output_size_per_class"],
            self.config["max_total_size"],
            self.config["yolo_iou_threshold"],
            self.config["yolo_score_threshold"],
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
        labels = np.asarray([self.class_labels[int(i)] for i in classes])

        return bboxes, labels, scores
