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
Object detection class using yolo single label model
to find license plate object bboxes
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


class Detector:  # pylint: disable=too-many-instance-attributes
    """Object detection class using yolo model to find object bboxes"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model_dir: Path,
        class_names: List[str],
        model_type: str,
        model_file: Dict[str, str],
        max_output_size_per_class: int,
        max_total_size: int,
        input_size: int,
        iou_threshold: float,
        score_threshold: float,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        self.class_names = class_names
        self.model_type = model_type
        self.model_path = model_dir / model_file[self.model_type]

        self.max_output_size_per_class = max_output_size_per_class
        self.max_total_size = max_total_size
        self.input_size = (input_size, input_size)
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

        self.yolo = self._create_yolo_model()

    def predict_object_bbox_from_image(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detects all license plate objects' bounding box from one image

        Args:
            image (np.ndarray): Input image.

        Return:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): A tuple containing the
            following arrays:
            - boxes: An array of bounding box with definition like
                (x1, y1, x2, y2), in a coordinate system with origin point in
                the top-left corner
            - labels: An array of class labels.
            - scores: An array of confidence scores.
        """
        # Use TF2 .pb saved model format for inference
        image_data = cv2.resize(image, self.input_size)
        image_data = image_data / 255.0

        image_data = np.asarray([image_data]).astype(np.float32)
        pred_bbox = self.yolo(tf.constant(image_data))
        for _, value in pred_bbox.items():
            pred_conf = value[:, :, 4:]
            boxes = value[:, :, 0:4]

        # performs nms using model's predictions
        bboxes, scores, classes, valid_dets = tf.image.combined_non_max_suppression(
            tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])
            ),
            self.max_output_size_per_class,
            self.max_total_size,
            self.iou_threshold,
            self.score_threshold,
        )
        classes = classes.numpy()[0]
        classes = classes[: valid_dets[0]]
        bboxes = bboxes.numpy()[0]
        bboxes = bboxes[: valid_dets[0]]
        scores = scores.numpy()[0]
        scores = scores[: valid_dets[0]]

        bboxes[:, [0, 1]] = bboxes[:, [1, 0]]  # swapping x and y axes
        bboxes[:, [2, 3]] = bboxes[:, [3, 2]]

        # scaling of bboxes if v4tiny model is used
        if self.model_type == "v4tiny":
            bboxes = self.scale_bboxes(bboxes, 0.75)

        # update the labels names of the object detected
        labels = np.asarray([self.class_names[int(i)] for i in classes])

        return bboxes, labels, scores

    def _create_yolo_model(self) -> Callable:
        """Creates yolo model for license plate detection."""
        self.logger.info(
            "Yolo model loaded with following configs:\n\t"
            f"Model type: {self.model_type},\n\t"
            f"Input resolution: {self.input_size},\n\t"
            f"IOU threshold: {self.iou_threshold},\n\t"
            f"Score threshold: {self.score_threshold}"
        )

        return self._load_yolo_weights()

    def _load_yolo_weights(self) -> Callable:
        self.model = tf.saved_model.load(
            str(self.model_path), tags=[tag_constants.SERVING]
        )
        return self.model.signatures["serving_default"]

    @staticmethod
    def scale_bboxes(bboxes: np.ndarray, scale_factor: float) -> np.ndarray:
        """Scales the width and height of bboxes from v4tiny.

        After converting the model from .cfg and .weight file format (Alexey's
        Darknet repo) to tf model, bboxes are bigger. So downscaling is
        required for a better fit.
        """
        outputs = np.empty_like(bboxes)
        center_x = (bboxes[:, 0] + bboxes[:, 2]) / 2
        center_y = (bboxes[:, 1] + bboxes[:, 3]) / 2
        outputs[:, 0] = center_x - (bboxes[:, 2] - bboxes[:, 0]) / 2 * scale_factor
        outputs[:, 1] = center_y - (bboxes[:, 3] - bboxes[:, 1]) / 2 * scale_factor
        outputs[:, 2] = center_x + (bboxes[:, 2] - bboxes[:, 0]) / 2 * scale_factor
        outputs[:, 3] = center_y + (bboxes[:, 3] - bboxes[:, 1]) / 2 * scale_factor
        return outputs
