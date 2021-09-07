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

import os
import logging
from typing import Dict, Any, Tuple

import numpy as np
import tensorflow as tf

from .graph_functions import load_graph


class Detector:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.root_dir = config['root']
        self.min_size = self.config['mtcnn_min_size'] 
        self.threshold = self.config['mtcnn_threshold'] 
        self.factors = self.config['mtcnn_factors']

        self.mtcnn = self._create_mtcnn_model()

    def _create_mtcnn_model(self) -> tf.keras.Model:
        model_type = self.config['model_type']
        model_path = os.path.join(self.root_dir, self.config['graph_files'][model_type])

        return self._load_mtcnn_graph(model_path)

    def _load_mtcnn_graph(self, filepath: str) -> tf.compat.v1.GraphDef:
        model_path = os.path.join(filepath)
        if os.path.isfile(model_path):
            return load_graph(model_path)
        raise ValueError('Graph file does not exist. Please check that '
                         '%s exists' % model_path)

    def predict_bbox_landmarks(self, image: np.ndarray) -> Tuple[np.ndarray,
                                                                 np.ndarray,
                                                                 np.ndarray]:
        # 1. process inputs
        image = self._process_image(image)
        min_size, threshold, factors = self._process_params()

        # 2. evaluate image
        bboxes, scores, landmarks = self.mtcnn(image, min_size, threshold, factors)

        # 3. process outputs
        bboxes, scores, landmarks = self._process_outputs(image, bboxes, scores, landmarks)

        # 4. create bbox_labels
        classes = [0]*len(bboxes)

        return bboxes, scores, landmarks, classes

    def _process_image(self, image: np.ndarray) -> tf.Tensor:
        image = image.astype(np.float32)
        image = tf.convert_to_tensor(image)
        return image

    def _process_params(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        min_size = tf.convert_to_tensor(float(self.min_size))
        threshold = tf.convert_to_tensor(float(self.threshold))
        factors = [float(integer) for integer in self.factors]
        factors = tf.convert_to_tensor(factors)
        return min_size, threshold, factors

    def _process_outputs(self, image: np.ndarray, bboxes: tf.Tensor, 
                         scores: tf.Tensor, landmarks: tf.Tensor) -> Tuple[np.ndarray,
                                                                           np.ndarray,
                                                                           np.ndarray]:
        bboxes, scores, landmarks = bboxes.numpy(), scores.numpy(), landmarks.numpy()
        bboxes[:,[0, 1]] = bboxes[:,[1, 0]]
        bboxes[:,[2, 3]] = bboxes[:,[3, 2]]
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]]/image.shape[1]
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]]/image.shape[0]
        return bboxes, scores, landmarks







