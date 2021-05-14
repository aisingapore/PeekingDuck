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
import os
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import tensorflow as tf
from peekingduck.utils.graph_functions import load_graph
from peekingduck.pipeline.nodes.model.efficientdet_d04.efficientdet_files.model import efficientdet
from peekingduck.pipeline.nodes.model.efficientdet_d04.efficientdet_files.utils.model_process \
    import preprocess_image, postprocess_boxes

GRAPH_MODE = True


class Detector:
    """Detector class to handle detection of bboxes for efficientdet
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.root_dir = config['root']

        self.effdet = self._create_effdet_model()

    def _create_effdet_model(self) -> tf.keras.Model:
        self.model_type = self.config['model_type']
        if GRAPH_MODE:
            graph_path = os.path.join(self.root_dir, self.config['graph_files'][self.model_type])
            model_nodes = self.config['MODEL_NODES']
            model = load_graph(
                graph_path, inputs=model_nodes['inputs'], outputs=model_nodes['outputs'])
            self.logger.info(
                'Efficientdet graph model loaded with following configs:'
                'Model type: D%s, '
                'Score Threshold: %s, ',
                self.model_type, self.config['score_threshold'])
            return model
        # For keras model
        _, model = efficientdet(phi=self.model_type,
                                num_classes=self.config['num_classes'],
                                score_threshold=self.config['score_threshold'])
        model_path = os.path.join(self.root_dir, self.config['model_files'][self.model_type])
        model.load_weights(model_path, by_name=True)
        self.logger.info(
            'Efficientdet keras model loaded with following configs:'
            'Model type: D%s, '
            'Score Threshold: %s, ',
            self.model_type, self.config['score_threshold'])
        return model

    @staticmethod
    def preprocess(image: List[List[float]],
                   image_size: int) -> Tuple[List[List[float]], float]:
        """Preprocessing function for efficientdet

        Args:
            image (np.array): image in numpy array
            image_size (int): image size as defined in efficientdet config

        Returns:
            image (np.array): the preprocessed image
            scale (float): the scale the image was resized to
        """
        image, scale = preprocess_image(image, image_size=image_size)
        return image, scale

    def postprocess(self, network_output: Tuple[np.ndarray, np.ndarray, np.ndarray],
                    scale: float,
                    img_shape: List[int],
                    detect_ids: List[int]) -> Tuple[List, List, List]:
        """Postprocessing of detected bboxes for efficientdet

        Args:
            network_output (list): list of boxes, scores and labels from network
            scale (float): scale the image was resized to
            img_shape (list): height of original image
            detect_ids (list): list of label ids to be detected

        Returns:
            boxes (np.array): postprocessed array of detected bboxes
            scores (np.array): postprocessed array of scores
            labels (np.array): postprocessed array of labels
        """
        img_h, img_w = img_shape
        boxes, scores, labels = network_output
        boxes = postprocess_boxes(boxes, scale, img_h, img_w)

        indices = np.where(scores[:] > self.config['score_threshold'])[0]

        # select those detections
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]

        detect_filter = np.where(np.isin(labels, detect_ids))
        boxes = boxes[detect_filter]
        labels = labels[detect_filter]
        scores = scores[detect_filter]

        return boxes, labels, scores

    def predict_bbox_from_image(self,
                                image: np.ndarray,
                                detect_ids: List[int]) -> Tuple[List, List, List]:
        """Efficientdet bbox prediction function

        Args:
            image (np.array): image in numpy array
            detect_ids (list): list of label ids to be detected

        Returns:
            boxes (np.array): array of detected bboxes
            scores (np.array): array of scores
            labels (np.array): array of labels
        """
        img_shape = image.shape[:2]

        image_size = self.config['size'][self.model_type]
        image, scale = self.preprocess(image, image_size=image_size)

        # run network
        if GRAPH_MODE:
            graph_input = tf.convert_to_tensor(np.expand_dims(image, axis=0), dtype=tf.float32)
            boxes, scores, labels = self.effdet(x=graph_input)
            network_output = np.squeeze(boxes.numpy()), np.squeeze(
                scores.numpy()), np.squeeze(labels.numpy())
        else:
            boxes, scores, labels = self.effdet.predict_on_batch([np.expand_dims(image, axis=0)])
            network_output = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)

        boxes, labels, scores = self.postprocess(network_output, scale, img_shape, detect_ids)

        return boxes, labels, scores
