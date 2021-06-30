# Modifications copyright 2021 AI Singapore

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Original copyright (c) 2019 Zihao Zhang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Core Yolo model files
"""

from typing import Union, List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
)
from tensorflow.keras.regularizers import l2
from .batch_norm import BatchNormalization

# pylint: disable=redundant-keyword-arg, no-value-for-parameter, unexpected-keyword-arg, invalid-name, too-many-locals

YOLO_ANCHORS = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198),
                         (373, 326)], np.float32) / 416
YOLO_ANCHOR_MASKS = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

YOLO_TINY_ANCHORS = np.array([(10, 14), (23, 27), (37, 58), (81, 82),
                              (135, 169), (344, 319)], np.float32) / 416
YOLO_TINY_ANCHOR_MASKS = np.array([[3, 4, 5], [0, 1, 2]])


def _darknet_conv(x: np.array, filters: int, size: int,
                  strides: int = 1, batch_norm: bool = True) -> tf.Tensor:
    """create 1 layer with [padding], conv2d, [bn and relu]"""
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters,
               kernel_size=size,
               strides=strides,
               padding=padding,
               use_bias=not batch_norm,
               kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x


def _darknet_residual(x: np.array, filters: int) -> tf.Tensor:
    """create 2 layers by given H(x) = F(x) + x"""
    prev = x
    x = _darknet_conv(x, filters // 2, 1)
    x = _darknet_conv(x, filters, 3)
    x = Add()([prev, x])
    return x


def _darknet_block(x: np.array, filters: int, blocks: int) -> tf.Tensor:
    """create (1 + 2 x blocks) layers"""
    x = _darknet_conv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = _darknet_residual(x, filters)
    return x


def _darknet(name: str = None) -> tf.keras.Model:
    """Create a Model with 52 layers

    This is different from classical Darknet, which shall be 53 layers,
    it can be calculated as below:
    1 + (1 + 2x1) + (1 + 2x2) + (1 + 2x8) + (1 + 2x8) + (1 + 2x4)

    Return:
        - tf.keras.Model with 3 outputs x_36, x_61 an x
    """
    x = inputs = Input([None, None, 3])
    x = _darknet_conv(x, 32, 3)
    x = _darknet_block(x, 64, 1)
    x = _darknet_block(x, 128, 2)  # skip connection
    x = x_36 = _darknet_block(x, 256, 8)  # skip connection
    x = x_61 = _darknet_block(x, 512, 8)
    x = _darknet_block(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def _darknet_tiny(name: str = None) -> tf.keras.Model:
    """create 7 layers, return a Model with 2 outputs x_8 and x"""
    x = inputs = Input([None, None, 3])
    x = _darknet_conv(x, 16, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = _darknet_conv(x, 32, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = _darknet_conv(x, 64, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = _darknet_conv(x, 128, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = x_8 = _darknet_conv(x, 256, 3)  # skip connection
    x = MaxPool2D(2, 2, 'same')(x)
    x = _darknet_conv(x, 512, 3)
    x = MaxPool2D(2, 1, 'same')(x)
    x = _darknet_conv(x, 1024, 3)
    return tf.keras.Model(inputs, (x_8, x), name=name)


def _yolo_conv(filters: int, name: str = None) -> tf.keras.Model:
    """create convolution layers

    It has [1] + 5 layers

    Return:
        - Model with 1 output x
    """
    def _yolo_conv_imp(x_in: Union[np.array, Tuple[int, int]]) -> tf.keras.Model:
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = _darknet_conv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = _darknet_conv(x, filters, 1)
        x = _darknet_conv(x, filters * 2, 3)
        x = _darknet_conv(x, filters, 1)
        x = _darknet_conv(x, filters * 2, 3)
        x = _darknet_conv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)

    return _yolo_conv_imp


def _yolo_conv_tiny(filters: int, name: str = None) -> tf.keras.Model:
    """create [1] layers, return a Model with 1 output x"""
    def _yolo_conv_tiny_imp(x_in: Union[np.array, Tuple[int, int]]) -> tf.keras.Model:
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = _darknet_conv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
            x = _darknet_conv(x, filters, 1)

        return Model(inputs, x, name=name)(x_in)

    return _yolo_conv_tiny_imp


def _yolo_output(filters: int, anchors: int, classes: int,
                 name: str = None) -> tf.keras.Model:
    """create 2 layers, return a model with 1 output x"""
    def _yolo_output_imp(x_in: np.array) -> tf.keras.Model:
        x = inputs = Input(x_in.shape[1:])
        x = _darknet_conv(x, filters * 2, 3)
        x = _darknet_conv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)

    return _yolo_output_imp


def _yolo_boxes(pred: tf.Tensor, anchors: int,
                classes: int) -> Tuple[List[np.array], List[str], List[float], List[np.array]]:
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1]
    box_xy, box_wh, objectness, class_probs = tf.split(pred,
                                                       (2, 2, 1, classes),
                                                       axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def _yolo_nms(outputs: List[tf.Tensor],
              classes: tf.Tensor) -> Tuple[List[np.array], List[float], List[str], List[int]]:
    """non-maximum suppression"""
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores,
                          (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=0.5,
        score_threshold=0.2)

    return boxes, scores, classes, valid_detections


def yolov3(size: int = None,
           channels: int = 3,
           classes: int = 80,
           training: bool = False) -> tf.keras.Model:
    """Create a yolov3 model

    This model has 76 layers, which can be calcualted below:
    1 darknet (52 layers) +
    3 yolo_conv (6 layers) +
    3 yolo_output (2 layers)

    Args:
        - size:     (int) input image size
        - channels: (int) channel for input image, for RBG it is 3
        - anchors:  (np.array) the boxes used to detect object, with a series of (x, y)
                    values, in this case it is 9 normalized (x, y) values.
        - masks:    (np.array) masks for anchors to decide which anchors will be used,
                    in this case it is 3 of (i, k, j) values.
        - classes:  (int) the number of classes of objects this Yolov3 can detect
        - training: (boolean) whether it is in training or not

    Returns:
        - (tensorflow.keras.Model) with inputs and outputs (3 items in outputs)
    """
    anchors = YOLO_ANCHORS
    masks = YOLO_ANCHOR_MASKS

    x = inputs = Input([size, size, channels])

    x_36, x_61, x = _darknet(name='yolo_darknet')(x)

    x = _yolo_conv(512, name='yolo_conv_0')(x)
    output_0 = _yolo_output(
        512, len(masks[0]), classes, name='yolo_output_0')(x)

    x = _yolo_conv(256, name='yolo_conv_1')((x, x_61))
    output_1 = _yolo_output(
        256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = _yolo_conv(128, name='yolo_conv_2')((x, x_36))
    output_2 = _yolo_output(
        128, len(masks[2]), classes, name='yolo_output_2')(x)

    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov3')

    boxes_0 = Lambda(lambda x: _yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: _yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: _yolo_boxes(x, anchors[masks[2]], classes),
                     name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda x: _yolo_nms(x, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')


def yolov3_tiny(size: int = None,
                channels: int = 3,
                classes: int = 80,
                training: bool = False) -> tf.keras.Model:
    """Create Yolov3 tiny model

    This model has 23 layers, which can be calcualted below:
    1 darknet_tiny (7 layers) +
    2 yolo_conv (6 layers) +
    2 yolo_output (2 layers)

    Input:
        - size:     (int) input image size
        - channels: (int) channel for input image, for RBG it is 3
        - anchors:  (np.array) the boxes used to detect object, with a series of (x, y)
                    values, in this case it is 9 of (x, y) values
        - masks:    (np.array) masks for anchors to decide which anchors will be used,
                    in this case it is 3 of (i, k, j) values
        - classes:  (int) the number of classes of objects this Yolov3 can detect
        - training: (boolean) whether it is in training or not

    Return:
        - (tensorflow.keras.Model) with inputs and outputs (2 items in outputs)
    """
    anchors = YOLO_TINY_ANCHORS
    masks = YOLO_TINY_ANCHOR_MASKS

    x = inputs = Input([size, size, channels])

    x_8, x = _darknet_tiny(name='yolo_darknet')(x)

    x = _yolo_conv_tiny(256, name='yolo_conv_0')(x)
    output_0 = _yolo_output(256, len(masks[0]), classes,
                            name='yolo_output_0')(x)

    x = _yolo_conv_tiny(128, name='yolo_conv_1')((x, x_8))
    output_1 = _yolo_output(128, len(masks[1]), classes,
                            name='yolo_output_1')(x)

    if training:
        return Model(inputs, (output_0, output_1), name='yolov3')

    boxes_0 = Lambda(lambda x: _yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: _yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    outputs = Lambda(lambda x: _yolo_nms(x, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3]))

    return Model(inputs, outputs, name='yolov3_tiny')
