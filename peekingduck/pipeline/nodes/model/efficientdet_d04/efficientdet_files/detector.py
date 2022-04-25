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
Detector class to handle detection of bboxes for efficientdet
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from peekingduck.pipeline.nodes.model.efficientdet_d04.efficientdet_files.model_process import (
    postprocess_boxes,
    preprocess_image,
)
from peekingduck.utils.graph_functions import load_graph


class Detector:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """Detector class to handle detection of bboxes for efficientdet"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model_dir: Path,
        class_names: Dict[int, str],
        detect_ids: List[int],
        model_type: int,
        num_classes: int,
        model_file: Dict[int, str],
        model_nodes: Dict[str, List[str]],
        image_size: Dict[int, int],
        score_threshold: float,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        self.class_names = class_names
        self.model_type = model_type
        self.num_classes = num_classes
        self.model_path = model_dir / model_file[self.model_type]
        self.model_nodes = model_nodes
        self.image_size = image_size[self.model_type]
        self.score_threshold = score_threshold

        self.detect_ids = detect_ids
        self.efficient_det = self._create_efficient_det_model()

    def predict_object_bbox_from_image(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Efficientdet bbox prediction function

        Args:
            image (np.ndarray): image in numpy array

        Returns:
            boxes (np.ndarray): array of detected bboxes
            labels (np.ndarray): array of labels
            scores (np.ndarray): array of scores
        """
        img_shape = image.shape[:2]
        image, scale = self._preprocess(image)

        # run network
        graph_input = tf.convert_to_tensor(
            np.expand_dims(image, axis=0), dtype=tf.float32
        )
        boxes, scores, labels = self.efficient_det(x=graph_input)
        network_output = (
            np.squeeze(boxes.numpy()),
            np.squeeze(scores.numpy()),
            np.squeeze(labels.numpy()),
        )

        boxes, labels, scores = self._postprocess(network_output, scale, img_shape)

        return boxes, labels, scores

    def _create_efficient_det_model(self) -> tf.keras.Model:
        model = load_graph(
            str(self.model_path),
            inputs=self.model_nodes["inputs"],
            outputs=self.model_nodes["outputs"],
        )
        self.logger.info(
            "EfficientDet model loaded with following configs:\n\t"
            f"Model type: D{self.model_type}\n\t"
            f"IDs being detected: {self.detect_ids}\n\t"
            f"Score threshold: {self.score_threshold}"
        )

        return model

    def _postprocess(
        self,
        network_output: Tuple[np.ndarray, np.ndarray, np.ndarray],
        scale: float,
        img_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Postprocessing of detected bboxes for efficientdet

        Args:
            network_output (list): list of boxes, scores and labels from network
            scale (float): scale the image was resized to
            img_shape (Tuple[int, int]): height of original image

        Returns:
            boxes (np.ndarray): postprocessed array of detected bboxes
            scores (np.ndarray): postprocessed array of scores
            labels (np.ndarray): postprocessed array of labels
        """
        img_h, img_w = img_shape
        boxes, scores, labels = network_output
        boxes = postprocess_boxes(boxes, scale, img_h, img_w)

        # Filter by confidence score
        score_filter = np.where(scores[:] > self.score_threshold)[0]
        boxes = boxes[score_filter]
        labels = labels[score_filter]
        scores = scores[score_filter]

        # Filter by detect ID
        detect_filter = np.where(np.isin(labels, self.detect_ids))
        boxes = boxes[detect_filter]
        labels = labels[detect_filter]
        scores = scores[detect_filter]

        if labels.size:
            labels = np.vectorize(self.class_names.get)(labels)
        return boxes, labels, scores

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Preprocessing function for efficientdet

        Args:
            image (np.ndarray): Image in numpy array.

        Returns:
            image (np.ndarray): the preprocessed image
            scale (float): the scale the image was resized to
        """
        return preprocess_image(image, self.image_size)
