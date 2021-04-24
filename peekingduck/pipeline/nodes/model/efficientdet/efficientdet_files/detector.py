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
import numpy as np
from .model import efficientdet
from .utils.model_process import preprocess_image, postprocess_boxes


class Detector:
    """Detector class to handle detection of bboxes for efficientdet
    """

    def __init__(self, config):
        self.config = config
        self.root_dir = config['root']

        self.effdet = self._create_effdet_model()

    def _create_effdet_model(self):
        self.model_type = self.config['model_type']
        if self.config['efficientdet_graph_mode']:
            raise NotImplementedError
        _, model = efficientdet(phi=self.model_type,
                                weighted_bifpn=self.config['weighted_bifpn'],
                                num_classes=self.config['num_classes'],
                                score_threshold=self.config['score_threshold'])
        model_path = os.path.join(self.root_dir, self.config['model_files'][self.model_type])
        model.load_weights(model_path, by_name=True)
        return model

    @staticmethod
    def preprocess(image, image_size):
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

    def postprocess(self, network_output, scale, img_shape, detect_ids):
        """Postprocessing of detected bboxes for efficientdet

        Args:
            network_output (list): list of boxes, scores and labels from network
            scale (float): scale the image was resized to
            img_h (int): height of original image
            img_w (int): width of original image
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

    def predict_bbox_from_image(self, image, detect_ids):
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
        boxes, scores, labels = self.effdet.predict_on_batch([np.expand_dims(image, axis=0)])
        network_output = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)

        boxes, labels, scores = self.postprocess(network_output, scale, img_shape, detect_ids)

        return boxes, labels, scores
