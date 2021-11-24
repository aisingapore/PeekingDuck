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
Object detection class using yolo model to detect human faces
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


class Detector:  # pylint: disable=too-few-public-methods
    """Object detection class using yolo model to find human faces"""

    def __init__(self, config: Dict[str, Any], model_dir: Path) -> None:
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.model_dir = model_dir
        self.class_labels = self._get_class_labels()
        self.yolo = self._create_yolo_model()

    def _get_class_labels(self) -> List[str]:
        classes_path = self.model_dir / self.config["weights"]["classes_file"]
        with open(classes_path, "rt", encoding="utf8") as file:
            class_labels = [c.strip() for c in file.readlines()]

        return class_labels

    def _create_yolo_model(self) -> cv2.dnn_Net:
        model_type = self.config["model_type"]
        model_path = (
            self.model_dir / self.config["weights"]["saved_model_subdir"][model_type]
        )
        model = tf.saved_model.load(str(model_path), tags=[tag_constants.SERVING])

        self.logger.info(
            "Yolo model loaded with following configs: \n\t"
            f"Model type: {self.config['model_type']}, \n\t"
            f"Input resolution: {self.config['size']}, \n\t"
            f"IDs being detected: {self.config['detect_ids']}, \n\t"
            f"Max detections per class: {self.config['max_output_size_per_class']}, \n\t"
            f"Max total detections: {self.config['max_total_size']}, \n\t"
            f"IOU threshold: {self.config['yolo_iou_threshold']}, \n\t"
            f"Score threshold: {self.config['yolo_score_threshold']}"
        )

        return model

    def predict_object_bbox_from_image(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts face bboxes, labels and scores

        Args:
            image (np.ndarray): image in numpy array

        Returns:
            bboxes (np.ndarray): numpy array of detected bboxes
            classes (np.ndarray): numpy array of class labels
            scores (np.ndarray): numpy array of confidence scores
        """
        # 1. process inputs
        image = self._process_image(image)

        # 2. evaluate image
        bboxes, scores, classes, nums = self._evaluate_image_by_yolo(image)

        # 3. clean up return
        bboxes, scores, classes = self._shrink_dimension_and_length(
            bboxes, scores, classes, nums
        )

        # 4. convert classes into class names
        classes = np.array([self.class_labels[int(i)] for i in classes])

        return bboxes, classes, scores

    def _process_image(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.config["size"], self.config["size"]))
        image = np.asarray([image]).astype(np.float32) / 255.0

        return image

    def _evaluate_image_by_yolo(
        self, image: np.ndarray
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Performs inference on a given input image.

        Args:
            image (np.ndarray): image in numpy array

        Returns:
            bboxes (tf.Tensor): EagerTensor object of detected bboxes
            scores (tf.Tensor): EagerTensor object of confidence scores
            classes (tf.Tensor): EagerTensor object of class labels
            nums (tf.Tensor): number of valid bboxes. Only nums[0] should be used.
            The rest are paddings.
        """
        infer = self.yolo.signatures["serving_default"]
        pred_bbox = infer(tf.constant(image))
        for value in pred_bbox.values():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        # performs nms using model's predictions
        bboxes, scores, classes, nums = tf.image.combined_non_max_suppression(
            max_output_size_per_class=self.config["max_output_size_per_class"],
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])
            ),
            score_threshold=self.config["yolo_score_threshold"],
            max_total_size=self.config["max_total_size"],
            iou_threshold=self.config["yolo_iou_threshold"],
        )

        return bboxes, scores, classes, nums

    def _shrink_dimension_and_length(
        self, bboxes: tf.Tensor, scores: tf.Tensor, classes: tf.Tensor, nums: tf.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        len0 = nums[0]

        classes = classes.numpy()[0]
        classes = classes[:len0]
        # only identify objects we are interested in
        mask1 = np.isin(classes, tuple(self.config["detect_ids"]))

        scores = scores.numpy()[0]
        scores = scores[:len0]
        scores = scores[mask1]

        bboxes = bboxes.numpy()[0]
        bboxes = bboxes[:len0]
        bboxes = bboxes[mask1]

        bboxes[:, [0, 1]] = bboxes[:, [1, 0]]
        bboxes[:, [2, 3]] = bboxes[:, [3, 2]]

        return bboxes, scores, classes
