# Copyright 2021 AI Singapore
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
#
# Code of this file is mostly forked from
# [@xuannianz](https://github.com/xuannianz))

"""
EfficientDet layers
"""

from typing import Any, Dict, List
from tensorflow import keras
import tensorflow as tf


class WBiFPNAdd(keras.layers.Layer):
    """Class for Weighted Bi-directional FPN
    """

    def __init__(self, epsilon: float = 1e-4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.weight = None

    def build(self, input_shape: List[tf.TensorShape]) -> None:
        num_in = len(input_shape)
        self.weight = self.add_weight(name=self.name,
                                      shape=(num_in,),
                                      initializer=keras.initializers.constant(1 / num_in),
                                      trainable=True,
                                      dtype=tf.float32)

    def call(self, inputs: List[tf.Tensor], **kwargs: Dict[str, Any]) -> tf.Tensor:
        weight = keras.activations.relu(self.weight)
        x_in = tf.reduce_sum([weight[i] * inputs[i] for i in range(len(inputs))], axis=0)
        x_in = x_in / (tf.reduce_sum(weight) + self.epsilon)
        return x_in

    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> tf.TensorShape:
        return input_shape[0]

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon
        })
        return config


def bbox_transform_inv(boxes: tf.Tensor,
                       deltas: tf.Tensor,
                       scale_factors: List[float] = None) -> tf.Tensor:
    """Helper function to transform bboxes using offsets

    Args:
        boxes : detected bboxes]
        deltas : bbox offsets
        scale_factors : List of scales for each dimension

    Returns:
        tf.Tensor: bboxes in xmin, ymin, xmax, ymax format
    """

    wt_a = boxes[..., 2] - boxes[..., 0]
    ht_a = boxes[..., 3] - boxes[..., 1]
    t_y, t_x, t_h, t_w = deltas[..., 0], deltas[..., 1], deltas[..., 2], deltas[..., 3]
    if scale_factors:
        t_y *= scale_factors[0]
        t_x *= scale_factors[1]
        t_h *= scale_factors[2]
        t_w *= scale_factors[3]
    width = tf.exp(t_w) * wt_a
    height = tf.exp(t_h) * ht_a
    center_y = t_y * ht_a + (boxes[..., 1] + boxes[..., 3]) / 2
    center_x = t_x * wt_a + (boxes[..., 0] + boxes[..., 2]) / 2
    top_left = center_x - width / 2., center_y - height / 2.
    btm_right = center_x + width / 2., center_y + height / 2.
    # return tf.stack([xmin, ymin, xmax, ymax], axis=-1)
    return tf.stack([top_left[0], top_left[1], btm_right[0], btm_right[1]], axis=-1)


class ClipBoxes(keras.layers.Layer):
    """ClipBoxes class to limit the value of bbox coordinates to image height and width
    """

    def call(self, inputs: List[tf.Tensor], **kwargs: Dict[str, Any]) -> tf.Tensor:
        image, boxes = inputs
        shape = keras.backend.cast(keras.backend.shape(image), keras.backend.floatx())
        height = shape[1]
        width = shape[2]
        x_1 = tf.clip_by_value(boxes[:, :, 0], 0, width - 1)
        y_1 = tf.clip_by_value(boxes[:, :, 1], 0, height - 1)
        x_2 = tf.clip_by_value(boxes[:, :, 2], 0, width - 1)
        y_2 = tf.clip_by_value(boxes[:, :, 3], 0, height - 1)

        return keras.backend.stack([x_1, y_1, x_2, y_2], axis=2)

    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> tf.TensorShape:
        return input_shape[1]


class RegressBoxes(keras.layers.Layer):
    """RegressBoxes class to compute actual bbox coordinate using anchors and offsets
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def call(self, inputs: List[tf.Tensor], **kwargs: Dict[str, Any]) -> tf.Tensor:
        anchors, regression = inputs
        return bbox_transform_inv(anchors, regression)

    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> tf.TensorShape:
        return input_shape[0]

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        return config


def filter_detections(  # pylint: disable=too-many-arguments, too-many-locals
        boxes: tf.Tensor,
        classification: tf.Tensor,
        alphas: tf.Tensor = None,
        ratios: tf.Tensor = None,
        class_specific_filter: bool = True,
        nms: bool = True,
        score_threshold: float = 0.01,
        max_detections: int = 100,
        nms_threshold: float = 0.5,
        detect_quadrangle: bool = False,
) -> List[tf.Tensor]:
    """
    Filter detections using the boxes and classification values.

    Args
        boxes: Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        classification: Tensor of shape (num_boxes, num_classes) containing classification scores.
        other: List of tensors (num_boxes, ...) to filter the boxes and classification scores.
        class_specific_filter: Whether to filter per class, or take best scoring class and filter.
        nms: Flag to enable/disable non maximum suppression.
        score_threshold: Threshold used to prefilter the boxes with.
        max_detections: Maximum number of detections to keep.
        nms_threshold: Threshold for the IoU value to determine when a box should be suppressed.

    Returns
        A list of [boxes, scores, labels, other[0], other[1], ...].
        boxes (max_detections, 4) and contains the (x1, y1, x2, y2) of non-suppressed boxes.
        scores (max_detections,) and contains the scores of the predicted class.
        labels (max_detections,) and contains the predicted label.
        other[i] is shaped (max_detections, ...) and contains the filtered other[i] data.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """

    def _filter_detections(scores_: tf.Tensor, labels_: tf.Tensor) -> tf.Tensor:
        # threshold based on score
        # (num_score_keeps, 1)
        indices_ = tf.where(keras.backend.greater(scores_, score_threshold))

        if nms:
            # (num_score_keeps, 4)
            filtered_boxes = tf.gather_nd(boxes, indices_)
            # In [4]: scores = np.array([0.1, 0.5, 0.4, 0.2, 0.7, 0.2])
            # In [5]: tf.greater(scores, 0.4)
            # Out[5]: <tf.Tensor: id=2, shape=(6,), dtype=bool,
            # numpy=array([False,  True, False, False,  True, False])>
            # In [6]: tf.where(tf.greater(scores, 0.4))
            # Out[6]:
            # <tf.Tensor: id=7, shape=(2, 1), dtype=int64, numpy=
            # array([[1],
            #        [4]])>
            #
            # In [7]: tf.gather(scores, tf.where(tf.greater(scores, 0.4)))
            # Out[7]:
            # <tf.Tensor: id=15, shape=(2, 1), dtype=float64, numpy=
            # array([[0.5],
            #        [0.7]])>
            filtered_scores = keras.backend.gather(scores_, indices_)[:, 0]

            # perform NMS
            # filtered_boxes = tf.concat([filtered_boxes[..., 1:2], filtered_boxes[..., 0:1],
            #                             filtered_boxes[..., 3:4], filtered_boxes[..., 2:3]],
            # axis=-1)
            nms_indices = tf.image.non_max_suppression(filtered_boxes, filtered_scores,
                                                       max_output_size=max_detections,
                                                       iou_threshold=nms_threshold)

            # filter indices based on NMS
            # (num_score_nms_keeps, 1)
            indices_ = keras.backend.gather(indices_, nms_indices)

        # add indices to list of all indices
        # (num_score_nms_keeps, )
        labels_ = tf.gather_nd(labels_, indices_)
        # (num_score_nms_keeps, 2)
        indices_ = keras.backend.stack([indices_[:, 0], labels_], axis=1)

        return indices_

    if class_specific_filter:
        all_indices = []
        # perform per class filtering
        for category in range(int(classification.shape[1])):
            scores = classification[:, category]
            labels = category * tf.ones((keras.backend.shape(scores)[0],), dtype='int64')
            all_indices.append(_filter_detections(scores, labels))

        # concatenate indices to single tensor
        # (concatenated_num_score_nms_keeps, 2)
        indices = keras.backend.concatenate(all_indices, axis=0)
    else:
        scores = keras.backend.max(classification, axis=1)
        labels = keras.backend.argmax(classification, axis=1)
        indices = _filter_detections(scores, labels)

    # select top k
    scores = tf.gather_nd(classification, indices)
    labels = indices[:, 1]
    scores, top_indices = tf.nn.top_k(scores, k=keras.backend.minimum(
        max_detections, keras.backend.shape(scores)[0]))

    # filter input using the final set of indices
    indices = keras.backend.gather(indices[:, 0], top_indices)
    boxes = keras.backend.gather(boxes, indices)
    labels = keras.backend.gather(labels, top_indices)

    # zero pad the outputs
    pad_size = keras.backend.maximum(0, max_detections - keras.backend.shape(scores)[0])
    boxes = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores = tf.pad(scores, [[0, pad_size]], constant_values=-1)
    labels = tf.pad(labels, [[0, pad_size]], constant_values=-1)
    labels = keras.backend.cast(labels, 'int32')

    # set shapes, since we know what they are
    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])

    if detect_quadrangle:
        alphas = keras.backend.gather(alphas, indices)
        ratios = keras.backend.gather(ratios, indices)
        alphas = tf.pad(alphas, [[0, pad_size], [0, 0]], constant_values=-1)
        ratios = tf.pad(ratios, [[0, pad_size]], constant_values=-1)
        alphas.set_shape([max_detections, 4])
        ratios.set_shape([max_detections])
        return [boxes, scores, alphas, ratios, labels]

    return [boxes, scores, labels]


class FilterDetections(keras.layers.Layer):
    """
    Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(  # pylint: disable=too-many-arguments
            self,
            nms: bool = True,
            class_specific_filter: bool = True,
            nms_threshold: float = 0.5,
            score_threshold: float = 0.01,
            max_detections: int = 100,
            parallel_iterations: int = 32,
            detect_quadrangle: bool = False,
            **kwargs: Any) -> None:
        """
        Filters detections using score threshold, NMS and selecting the top-k detections.

        Args
            nms: Flag to enable/disable NMS.
            class_specific_filter: To filter per class, or take the best scoring class and filter.
            nms_threshold: Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold: Threshold used to prefilter the boxes with.
            max_detections: Maximum number of detections to keep.
            parallel_iterations: Number of batch items to process in parallel.
        """
        self.nms = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.parallel_iterations = parallel_iterations
        self.detect_quadrangle = detect_quadrangle
        super().__init__(**kwargs)

    def call(self, inputs: List[tf.Tensor], **kwargs: Dict[str, Any]) -> tf.Tensor:
        """
        Constructs the NMS graph.

        Args
            inputs : List of [boxes, classification, other[0], other[1], ...] tensors.
        """
        boxes = inputs[0]
        classification = inputs[1]
        if self.detect_quadrangle:
            alphas = inputs[2]
            ratios = inputs[3]

        # wrap nms with our parameters
        def _filter_detections(args: List[tf.Tensor]) -> List[tf.Tensor]:
            boxes_ = args[0]
            classification_ = args[1]
            alphas_ = args[2] if self.detect_quadrangle else None
            ratios_ = args[3] if self.detect_quadrangle else None

            return filter_detections(
                boxes_,
                classification_,
                alphas_,
                ratios_,
                nms=self.nms,
                class_specific_filter=self.class_specific_filter,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
                nms_threshold=self.nms_threshold,
                detect_quadrangle=self.detect_quadrangle,
            )

        # call filter_detections on each batch item
        if self.detect_quadrangle:
            outputs = tf.map_fn(
                _filter_detections,
                elems=[boxes, classification, alphas, ratios],
                dtype=['float32', 'float32', 'float32', 'float32', 'int32'],
                parallel_iterations=self.parallel_iterations
            )
        else:
            outputs = tf.map_fn(
                _filter_detections,
                elems=[boxes, classification],
                dtype=['float32', 'float32', 'int32'],
                parallel_iterations=self.parallel_iterations
            )

        return outputs

    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> tf.TensorShape:
        """
        Computes the output shapes given the input shapes.

        Args
            input_shape : List of input shapes [boxes, classification].

        Returns
            List of tuples representing the output shapes:
            [filtered_boxes.shape, filtered_scores.shape, filtered_labels.shape,
            filtered_other[0].shape, filtered_other[1].shape, ...]
        """
        if self.detect_quadrangle:
            return [
                (input_shape[0][0], self.max_detections, 4),
                (input_shape[1][0], self.max_detections),
                (input_shape[1][0], self.max_detections, 4),
                (input_shape[1][0], self.max_detections),
                (input_shape[1][0], self.max_detections),
            ]

        return [
            (input_shape[0][0], self.max_detections, 4),
            (input_shape[1][0], self.max_detections),
            (input_shape[1][0], self.max_detections),
        ]

    def compute_mask(self, inputs: List[tf.Tensor], mask: tf.Tensor = None) -> tf.Tensor:
        """
        This is required in Keras when there is more than 1 output.
        """
        return (len(inputs) + 1) * [None]

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super().get_config()
        config.update({
            'nms': self.nms,
            'class_specific_filter': self.class_specific_filter,
            'nms_threshold': self.nms_threshold,
            'score_threshold': self.score_threshold,
            'max_detections': self.max_detections,
            'parallel_iterations': self.parallel_iterations,
        })

        return config
