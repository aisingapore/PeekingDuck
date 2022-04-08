# Copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Object detection class using yolo model to find object bboxes
"""


import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import tensorflow as tf

from peekingduck.utils.graph_functions import load_graph


class Detector:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """Object detection class using YOLO model to find object bboxes"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model_dir: Path,
        class_names: List[str],
        detect_ids: List[int],
        model_type: str,
        model_file: Dict[str, str],
        model_nodes: Dict[str, List[str]],
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
        self.model_nodes = model_nodes

        self.max_output_size_per_class = max_output_size_per_class
        self.max_total_size = max_total_size
        self.input_size = (input_size, input_size)
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

        self.detect_ids = detect_ids
        self.yolo = self._create_yolo_model()

    def predict_object_bbox_from_image(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect all objects' bounding box from one image

        Args:
            image (np.ndarray): input image

        Return:
            boxes (np.array): an array of bounding box with definition like
                (x1, y1, x2, y2), in a coordinate system with original point in
                the top-left corner
        """
        image = self._preprocess(image)

        pred = self.yolo(image)[-1]

        bboxes, scores, classes = self._postprocess(pred[:, :, :4], pred[:, :, 4:])
        labels = np.array([self.class_names[int(i)] for i in classes])

        return bboxes, labels, scores

    def _create_yolo_model(self) -> Callable:
        """Creates YOLO model for human detection."""
        self.logger.info(
            "YOLO model loaded with following configs: \n\t"
            f"Model type: {self.model_type}, \n\t"
            f"Input resolution: {self.input_size}, \n\t"
            f"IDs being detected: {self.detect_ids} \n\t"
            f"Max detections per class: {self.max_output_size_per_class}, \n\t"
            f"Max total detections: {self.max_total_size}, \n\t"
            f"IOU threshold: {self.iou_threshold}, \n\t"
            f"Score threshold: {self.score_threshold}"
        )

        return self._load_yolo_weights()

    def _load_yolo_weights(self) -> Callable:
        """When loading a graph model, you need to explicitly state the input
        and output nodes of the graph. It is usually x:0 for input and Identity:0
        for outputs, depending on how many output nodes you have.
        """
        if not self.model_path.is_file():
            raise ValueError(
                f"Graph file does not exist. Please check that {self.model_path} exists"
            )
        return load_graph(
            str(self.model_path),
            inputs=self.model_nodes["inputs"],
            outputs=self.model_nodes["outputs"],
        )

    def _postprocess(
        self, pred_boxes: tf.Tensor, pred_scores: tf.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bboxes, scores, classes, valid_dets = tf.image.combined_non_max_suppression(
            tf.reshape(pred_boxes, (tf.shape(pred_boxes)[0], -1, 1, 4)),
            tf.reshape(
                pred_scores, (tf.shape(pred_scores)[0], -1, tf.shape(pred_scores)[-1])
            ),
            self.max_output_size_per_class,
            self.max_total_size,
            self.iou_threshold,
            self.score_threshold,
        )
        num_valid = valid_dets[0]

        classes = classes.numpy()[0]
        classes = classes[:num_valid]
        # only identify objects we are interested in
        mask = np.isin(classes, self.detect_ids)
        classes = classes[mask]

        scores = scores.numpy()[0]
        scores = scores[:num_valid]
        scores = scores[mask]

        bboxes = bboxes.numpy()[0]
        bboxes = bboxes[:num_valid]
        bboxes = bboxes[mask]

        # swapping x and y axes
        bboxes[:, [0, 1]] = bboxes[:, [1, 0]]
        bboxes[:, [2, 3]] = bboxes[:, [3, 2]]

        return bboxes, scores, classes

    def _preprocess(self, image: np.ndarray) -> tf.Tensor:
        processed_image = tf.convert_to_tensor(image.astype(np.float32))
        processed_image = tf.expand_dims(processed_image, 0)
        processed_image = tf.image.resize(processed_image, self.input_size) / 255.0

        return processed_image
