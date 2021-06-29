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
EfficientDet models
"""

from typing import Any, Callable, Tuple, List
from functools import reduce

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, initializers, models

from peekingduck.pipeline.nodes.model.efficientdet_d04.efficientdet_files.layers \
    import ClipBoxes, RegressBoxes, FilterDetections, WBiFPNAdd
from peekingduck.pipeline.nodes.model.efficientdet_d04.efficientdet_files.initializers \
    import PriorProbability
from peekingduck.pipeline.nodes.model.efficientdet_d04.efficientdet_files.tfkeras \
    import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, \
    EfficientNetB4, EfficientNetB5, EfficientNetB6
from peekingduck.pipeline.nodes.model.efficientdet_d04.efficientdet_files.utils.anchors \
    import anchors_for_shape

# pylint: disable=too-many-locals, too-many-statements, too-many-ancestors,too-many-instance-attributes, too-many-arguments

w_bifpns = [64, 88, 112, 160, 224, 288, 384]
d_bifpns = [3, 4, 5, 6, 7, 7, 8]
d_heads = [3, 3, 3, 4, 4, 4, 5]
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]
backbones = [EfficientNetB0, EfficientNetB1, EfficientNetB2,
             EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6]

MOMENTUM = 0.997
EPSILON = 1e-4


def separable_conv_block(num_channels: int, kernel_size: int, strides: int, name: str) -> Callable:
    """Separable conv block helper function"""
    f_1 = layers.SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides,
                                 padding='same', use_bias=True, name=f'{name}/conv')
    f_2 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{name}/bn')
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f_1, f_2))


def conv_block(num_channels: int, kernel_size: int, strides: int, name: str) -> Callable:
    """Conv block helper function"""
    f_1 = layers.Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                        use_bias=True, name='{}_conv'.format(name))
    f_2 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name='{}_bn'.format(name))
    f_3 = layers.ReLU(name='{}_relu'.format(name))
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f_1, f_2, f_3))


def build_wbi_fpn(features: List[tf.Tensor],
                  num_channels: int,
                  i_d: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Function to build weighted bi-directional fpn"""
    if i_d == 0:
        _, _, c_3, c_4, c_5 = features
        p3_in = c_3
        p4_in = c_4
        p5_in = c_5
        p6_in = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                              name='resample_p6/conv2d')(c_5)
        p6_in = layers.BatchNormalization(
            momentum=MOMENTUM, epsilon=EPSILON, name='resample_p6/bn')(p6_in)
        p6_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same',
                                    name='resample_p6/maxpool')(p6_in)
        p7_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same',
                                    name='resample_p7/maxpool')(p6_in)
        p7_u = layers.UpSampling2D()(p7_in)
        p6_td = WBiFPNAdd(name=f'fpn_cells/cell_{i_d}/fnode0/add')([p6_in, p7_u])
        p6_td = layers.Activation(tf.nn.swish)(p6_td)
        p6_td = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                     name=f'fpn_cells/cell_{i_d}/fnode0/op_after_combine5')(p6_td)
        p5_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{i_d}/fnode1/resample_0_2_6/conv2d')(p5_in)
        p5_in_1 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{i_d}/fnode1/'
                                            'resample_0_2_6/bn')(p5_in_1)
        p6_u = layers.UpSampling2D()(p6_td)
        p5_td = WBiFPNAdd(name=f'fpn_cells/cell_{i_d}/fnode1/add')([p5_in_1, p6_u])
        p5_td = layers.Activation(tf.nn.swish)(p5_td)
        p5_td = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                     name=f'fpn_cells/cell_{i_d}/fnode1/op_after_combine6')(p5_td)
        p4_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{i_d}/fnode2/resample_0_1_7/conv2d')(p4_in)
        p4_in_1 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{i_d}/fnode2/'
                                            'resample_0_1_7/bn')(p4_in_1)
        p5_u = layers.UpSampling2D()(p5_td)
        p4_td = WBiFPNAdd(name=f'fpn_cells/cell_{i_d}/fnode2/add')([p4_in_1, p5_u])
        p4_td = layers.Activation(tf.nn.swish)(p4_td)
        p4_td = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                     name=f'fpn_cells/cell_{i_d}/fnode2/op_after_combine7')(p4_td)
        p3_in = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                              name=f'fpn_cells/cell_{i_d}/fnode3/resample_0_0_8/conv2d')(p3_in)
        p3_in = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                          name=f'fpn_cells/cell_{i_d}/fnode3/'
                                          'resample_0_0_8/bn')(p3_in)
        p4_u = layers.UpSampling2D()(p4_td)
        p3_out = WBiFPNAdd(name=f'fpn_cells/cell_{i_d}/fnode3/add')([p3_in, p4_u])
        p3_out = layers.Activation(tf.nn.swish)(p3_out)
        p3_out = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                      name=f'fpn_cells/cell_{i_d}/fnode3/op_after_combine8')(p3_out)
        p4_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{i_d}/fnode4/resample_0_1_9/conv2d')(p4_in)
        p4_in_2 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{i_d}/fnode4/'
                                            'resample_0_1_9/bn')(p4_in_2)
        p3_d = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(p3_out)
        p4_out = WBiFPNAdd(name=f'fpn_cells/cell_{i_d}/fnode4/add')([p4_in_2, p4_td, p3_d])
        p4_out = layers.Activation(tf.nn.swish)(p4_out)
        p4_out = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                      name=f'fpn_cells/cell_{i_d}/fnode4/op_after_combine9')(p4_out)

        p5_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{i_d}/fnode5/resample_0_2_10/conv2d')(p5_in)
        p5_in_2 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{i_d}/fnode5/'
                                            'resample_0_2_10/bn')(p5_in_2)
        p4_d = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(p4_out)
        p5_out = WBiFPNAdd(name=f'fpn_cells/cell_{i_d}/fnode5/add')([p5_in_2, p5_td, p4_d])
        p5_out = layers.Activation(tf.nn.swish)(p5_out)
        p5_out = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                      name=f'fpn_cells/cell_{i_d}/fnode5/'
                                      'op_after_combine10')(p5_out)

        p5_d = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(p5_out)
        p6_out = WBiFPNAdd(name=f'fpn_cells/cell_{i_d}/fnode6/add')([p6_in, p6_td, p5_d])
        p6_out = layers.Activation(tf.nn.swish)(p6_out)
        p6_out = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                      name=f'fpn_cells/cell_{i_d}/fnode6/'
                                      'op_after_combine11')(p6_out)

        p6_d = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(p6_out)
        p7_out = WBiFPNAdd(name=f'fpn_cells/cell_{i_d}/fnode7/add')([p7_in, p6_d])
        p7_out = layers.Activation(tf.nn.swish)(p7_out)
        p7_out = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                      name=f'fpn_cells/cell_{i_d}/fnode7/'
                                      'op_after_combine12')(p7_out)

    else:
        p3_in, p4_in, p5_in, p6_in, p7_in = features
        p7_u = layers.UpSampling2D()(p7_in)
        p6_td = WBiFPNAdd(name=f'fpn_cells/cell_{i_d}/fnode0/add')([p6_in, p7_u])
        p6_td = layers.Activation(tf.nn.swish)(p6_td)
        p6_td = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                     name=f'fpn_cells/cell_{i_d}/fnode0/op_after_combine5')(p6_td)
        p6_u = layers.UpSampling2D()(p6_td)
        p5_td = WBiFPNAdd(name=f'fpn_cells/cell_{i_d}/fnode1/add')([p5_in, p6_u])
        p5_td = layers.Activation(tf.nn.swish)(p5_td)
        p5_td = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                     name=f'fpn_cells/cell_{i_d}/fnode1/op_after_combine6')(p5_td)
        p5_u = layers.UpSampling2D()(p5_td)
        p4_td = WBiFPNAdd(name=f'fpn_cells/cell_{i_d}/fnode2/add')([p4_in, p5_u])
        p4_td = layers.Activation(tf.nn.swish)(p4_td)
        p4_td = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                     name=f'fpn_cells/cell_{i_d}/fnode2/op_after_combine7')(p4_td)
        p4_u = layers.UpSampling2D()(p4_td)
        p3_out = WBiFPNAdd(name=f'fpn_cells/cell_{i_d}/fnode3/add')([p3_in, p4_u])
        p3_out = layers.Activation(tf.nn.swish)(p3_out)
        p3_out = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                      name=f'fpn_cells/cell_{i_d}/fnode3/op_after_combine8')(p3_out)
        p3_d = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(p3_out)
        p4_out = WBiFPNAdd(name=f'fpn_cells/cell_{i_d}/fnode4/add')([p4_in, p4_td, p3_d])
        p4_out = layers.Activation(tf.nn.swish)(p4_out)
        p4_out = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                      name=f'fpn_cells/cell_{i_d}/fnode4/op_after_combine9')(p4_out)

        p4_d = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(p4_out)
        p5_out = WBiFPNAdd(name=f'fpn_cells/cell_{i_d}/fnode5/add')([p5_in, p5_td, p4_d])
        p5_out = layers.Activation(tf.nn.swish)(p5_out)
        p5_out = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                      name=f'fpn_cells/cell_{i_d}/fnode5/'
                                      'op_after_combine10')(p5_out)

        p5_d = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(p5_out)
        p6_out = WBiFPNAdd(name=f'fpn_cells/cell_{i_d}/fnode6/add')([p6_in, p6_td, p5_d])
        p6_out = layers.Activation(tf.nn.swish)(p6_out)
        p6_out = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                      name=f'fpn_cells/cell_{i_d}/fnode6/'
                                      'op_after_combine11')(p6_out)

        p6_d = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(p6_out)
        p7_out = WBiFPNAdd(name=f'fpn_cells/cell_{i_d}/fnode7/add')([p7_in, p6_d])
        p7_out = layers.Activation(tf.nn.swish)(p7_out)
        p7_out = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                      name=f'fpn_cells/cell_{i_d}/fnode7/'
                                      'op_after_combine12')(p7_out)
    return p3_out, p4_td, p5_td, p6_td, p7_out


def build_bifpn(features: List[tf.Tensor],
                num_channels: int,
                i_d: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Function to build bi-directional fpn"""
    if i_d == 0:
        _, _, c_3, c_4, c_5 = features
        p3_in = c_3
        p4_in = c_4
        p5_in = c_5
        p6_in = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                              name='resample_p6/conv2d')(c_5)
        p6_in = layers.BatchNormalization(
            momentum=MOMENTUM, epsilon=EPSILON, name='resample_p6/bn')(p6_in)
        p6_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same',
                                    name='resample_p6/maxpool')(p6_in)
        p7_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same',
                                    name='resample_p7/maxpool')(p6_in)
        p7_u = layers.UpSampling2D()(p7_in)
        p6_td = layers.Add(name=f'fpn_cells/cell_{i_d}/fnode0/add')([p6_in, p7_u])
        p6_td = layers.Activation(tf.nn.swish)(p6_td)
        p6_td = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                     name=f'fpn_cells/cell_{i_d}/fnode0/op_after_combine5')(p6_td)
        p5_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{i_d}/fnode1/resample_0_2_6/conv2d')(p5_in)
        p5_in_1 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{i_d}/fnode1/'
                                            'resample_0_2_6/bn')(p5_in_1)
        p6_u = layers.UpSampling2D()(p6_td)
        p5_td = layers.Add(name=f'fpn_cells/cell_{i_d}/fnode1/add')([p5_in_1, p6_u])
        p5_td = layers.Activation(tf.nn.swish)(p5_td)
        p5_td = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                     name=f'fpn_cells/cell_{i_d}/fnode1/op_after_combine6')(p5_td)
        p4_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{i_d}/fnode2/resample_0_1_7/conv2d')(p4_in)
        p4_in_1 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{i_d}/fnode2/'
                                            'resample_0_1_7/bn')(p4_in_1)
        p5_u = layers.UpSampling2D()(p5_td)
        p4_td = layers.Add(name=f'fpn_cells/cell_{i_d}/fnode2/add')([p4_in_1, p5_u])
        p4_td = layers.Activation(tf.nn.swish)(p4_td)
        p4_td = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                     name=f'fpn_cells/cell_{i_d}/fnode2/op_after_combine7')(p4_td)
        p3_in = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                              name=f'fpn_cells/cell_{i_d}/fnode3/resample_0_0_8/conv2d')(p3_in)
        p3_in = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                          name=f'fpn_cells/cell_{i_d}/fnode3/'
                                          'resample_0_0_8/bn')(p3_in)
        p4_u = layers.UpSampling2D()(p4_td)
        p3_out = layers.Add(name=f'fpn_cells/cell_{i_d}/fnode3/add')([p3_in, p4_u])
        p3_out = layers.Activation(tf.nn.swish)(p3_out)
        p3_out = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                      name=f'fpn_cells/cell_{i_d}/fnode3/op_after_combine8')(p3_out)
        p4_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{i_d}/fnode4/resample_0_1_9/conv2d')(p4_in)
        p4_in_2 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{i_d}/fnode4/'
                                            'resample_0_1_9/bn')(p4_in_2)
        p3_d = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(p3_out)
        p4_out = layers.Add(name=f'fpn_cells/cell_{i_d}/fnode4/add')([p4_in_2, p4_td, p3_d])
        p4_out = layers.Activation(tf.nn.swish)(p4_out)
        p4_out = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                      name=f'fpn_cells/cell_{i_d}/fnode4/op_after_combine9')(p4_out)

        p5_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{i_d}/fnode5/resample_0_2_10/conv2d')(p5_in)
        p5_in_2 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{i_d}/fnode5/'
                                            'resample_0_2_10/bn')(p5_in_2)
        p4_d = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(p4_out)
        p5_out = layers.Add(name=f'fpn_cells/cell_{i_d}/fnode5/add')([p5_in_2, p5_td, p4_d])
        p5_out = layers.Activation(tf.nn.swish)(p5_out)
        p5_out = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                      name=f'fpn_cells/cell_{i_d}/fnode5/'
                                      'op_after_combine10')(p5_out)

        p5_d = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(p5_out)
        p6_out = layers.Add(name=f'fpn_cells/cell_{i_d}/fnode6/add')([p6_in, p6_td, p5_d])
        p6_out = layers.Activation(tf.nn.swish)(p6_out)
        p6_out = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                      name=f'fpn_cells/cell_{i_d}/fnode6/'
                                      'op_after_combine11')(p6_out)

        p6_d = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(p6_out)
        p7_out = layers.Add(name=f'fpn_cells/cell_{i_d}/fnode7/add')([p7_in, p6_d])
        p7_out = layers.Activation(tf.nn.swish)(p7_out)
        p7_out = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                      name=f'fpn_cells/cell_{i_d}/fnode7/'
                                      'op_after_combine12')(p7_out)

    else:
        p3_in, p4_in, p5_in, p6_in, p7_in = features
        p7_u = layers.UpSampling2D()(p7_in)
        p6_td = layers.Add(name=f'fpn_cells/cell_{i_d}/fnode0/add')([p6_in, p7_u])
        p6_td = layers.Activation(tf.nn.swish)(p6_td)
        p6_td = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                     name=f'fpn_cells/cell_{i_d}/fnode0/op_after_combine5')(p6_td)
        p6_u = layers.UpSampling2D()(p6_td)
        p5_td = layers.Add(name=f'fpn_cells/cell_{i_d}/fnode1/add')([p5_in, p6_u])
        p5_td = layers.Activation(tf.nn.swish)(p5_td)
        p5_td = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                     name=f'fpn_cells/cell_{i_d}/fnode1/op_after_combine6')(p5_td)
        p5_u = layers.UpSampling2D()(p5_td)
        p4_td = layers.Add(name=f'fpn_cells/cell_{i_d}/fnode2/add')([p4_in, p5_u])
        p4_td = layers.Activation(tf.nn.swish)(p4_td)
        p4_td = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                     name=f'fpn_cells/cell_{i_d}/fnode2/op_after_combine7')(p4_td)
        p4_u = layers.UpSampling2D()(p4_td)
        p3_out = layers.Add(name=f'fpn_cells/cell_{i_d}/fnode3/add')([p3_in, p4_u])
        p3_out = layers.Activation(tf.nn.swish)(p3_out)
        p3_out = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                      name=f'fpn_cells/cell_{i_d}/fnode3/op_after_combine8')(p3_out)
        p3_d = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(p3_out)
        p4_out = layers.Add(name=f'fpn_cells/cell_{i_d}/fnode4/add')([p4_in, p4_td, p3_d])
        p4_out = layers.Activation(tf.nn.swish)(p4_out)
        p4_out = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                      name=f'fpn_cells/cell_{i_d}/fnode4/op_after_combine9')(p4_out)

        p4_d = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(p4_out)
        p5_out = layers.Add(name=f'fpn_cells/cell_{i_d}/fnode5/add')([p5_in, p5_td, p4_d])
        p5_out = layers.Activation(tf.nn.swish)(p5_out)
        p5_out = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                      name=f'fpn_cells/cell_{i_d}/fnode5'
                                      '/op_after_combine10')(p5_out)

        p5_d = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(p5_out)
        p6_out = layers.Add(name=f'fpn_cells/cell_{i_d}/fnode6/add')([p6_in, p6_td, p5_d])
        p6_out = layers.Activation(tf.nn.swish)(p6_out)
        p6_out = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                      name=f'fpn_cells/cell_{i_d}/fnode6/'
                                      'op_after_combine11')(p6_out)

        p6_d = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(p6_out)
        p7_out = layers.Add(name=f'fpn_cells/cell_{i_d}/fnode7/add')([p7_in, p6_d])
        p7_out = layers.Activation(tf.nn.swish)(p7_out)
        p7_out = separable_conv_block(num_channels=num_channels, kernel_size=3, strides=1,
                                      name=f'fpn_cells/cell_{i_d}/fnode7/'
                                      'op_after_combine12')(p7_out)
    return p3_out, p4_td, p5_td, p6_td, p7_out


class BoxNet(models.Model):
    """Bbox regression network"""

    def __init__(self, width: int, depth: int, num_anchors: int = 9,
                 separable_conv: bool = True,
                 detect_quadrangle: bool = False,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.separable_conv = separable_conv
        self.detect_quadrangle = detect_quadrangle
        num_values = 9 if detect_quadrangle else 4
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }
        if separable_conv:
            kernel_initializer = {
                'depthwise_initializer': initializers.VarianceScaling(),
                'pointwise_initializer': initializers.VarianceScaling(),
            }
            options.update(kernel_initializer)
            self.convs = [layers.SeparableConv2D(filters=width,
                                                 name=f'{self.name}/box-{i}', **options) for i in
                          range(depth)]
            self.head = layers.SeparableConv2D(filters=num_anchors * num_values,
                                               name=f'{self.name}/box-predict', **options)
        else:
            kernel_initializer = {
                'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
            }
            options.update(kernel_initializer)
            self.convs = [layers.Conv2D(
                filters=width, name=f'{self.name}/box-{i}', **options) for i in range(depth)]
            self.head = layers.Conv2D(filters=num_anchors * num_values,
                                      name=f'{self.name}/box-predict', **options)
        self.bns = [
            [layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                       name=f'{self.name}/box-{i}-bn-{j}') for j in
             range(3, 8)]
            for i in range(depth)]
        self.relu = layers.Lambda(tf.nn.swish)
        self.reshape = layers.Reshape((-1, num_values))
        self.level = 0

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # pylint: disable=arguments-differ,
        feature, _ = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i][self.level](feature)
            feature = self.relu(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        self.level += 1
        return outputs


class ClassNet(models.Model):
    """Classification network"""

    def __init__(self, width: int, depth: int, num_classes: int = 20, num_anchors: int = 9,
                 separable_conv: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.separable_conv = separable_conv
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
        }
        if self.separable_conv:
            kernel_initializer = {
                'depthwise_initializer': initializers.VarianceScaling(),
                'pointwise_initializer': initializers.VarianceScaling(),
            }
            options.update(kernel_initializer)
            self.convs = [layers.SeparableConv2D(filters=width, bias_initializer='zeros',
                                                 name=f'{self.name}/class-{i}', **options)
                          for i in range(depth)]
            self.head = layers.SeparableConv2D(filters=num_classes * num_anchors,
                                               bias_initializer=PriorProbability(probability=0.01),
                                               name=f'{self.name}/class-predict', **options)
        else:
            kernel_initializer = {
                'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
            }
            options.update(kernel_initializer)
            self.convs = [layers.Conv2D(filters=width, bias_initializer='zeros',
                                        name=f'{self.name}/class-{i}', **options)
                          for i in range(depth)]
            self.head = layers.Conv2D(filters=num_classes * num_anchors,
                                      bias_initializer=PriorProbability(probability=0.01),
                                      name='class-predict', **options)
        self.bns = [
            [layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                       name=f'{self.name}/class-{i}-bn-{j}') for j in range(3, 8)]
            for i in range(depth)]

        self.relu = layers.Lambda(tf.nn.swish)
        self.reshape = layers.Reshape((-1, num_classes))
        self.activation = layers.Activation('sigmoid')
        self.level = 0

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # pylint: disable=arguments-differ,
        feature, _ = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i][self.level](feature)
            feature = self.relu(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        outputs = self.activation(outputs)
        self.level += 1
        return outputs


def efficientdet(phi: int, num_classes: int = 20, num_anchors: int = 9,
                 weighted_bifpn: bool = True, score_threshold: float = 0.01,
                 detect_quadrangle: bool = False, anchor_parameters: Any = None,
                 separable_conv: bool = True) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Function to build Efficientdet"""
    assert phi in range(7)
    input_size = image_sizes[phi]
    input_shape = (input_size, input_size, 3)
    image_input = layers.Input(input_shape)
    w_bifpn = w_bifpns[phi]
    d_bifpn = d_bifpns[phi]
    w_head = w_bifpn
    d_head = d_heads[phi]
    backbone_cls = backbones[phi]
    features = backbone_cls(input_tensor=image_input)
    if weighted_bifpn:
        fpn_features = features
        for i in range(d_bifpn):
            fpn_features = build_wbi_fpn(fpn_features, w_bifpn, i)
    else:
        fpn_features = features
        for i in range(d_bifpn):
            fpn_features = build_bifpn(fpn_features, w_bifpn, i)
    box_net = BoxNet(w_head, d_head, num_anchors=num_anchors, separable_conv=separable_conv,
                     detect_quadrangle=detect_quadrangle, name='box_net')
    class_net = ClassNet(w_head, d_head, num_classes=num_classes, num_anchors=num_anchors,
                         separable_conv=separable_conv, name='class_net')
    classification = [class_net([feature, i]) for i, feature in enumerate(fpn_features)]
    classification = layers.Concatenate(axis=1, name='classification')(classification)
    regression = [box_net([feature, i]) for i, feature in enumerate(fpn_features)]
    regression = layers.Concatenate(axis=1, name='regression')(regression)

    # model = models.Model(inputs=[image_input], outputs=[
    #                      classification, regression], name='efficientdet')

    # apply predicted regression to anchors
    anchors = anchors_for_shape((input_size, input_size), anchor_params=anchor_parameters)
    anchors_input = np.expand_dims(anchors, axis=0)
    boxes = RegressBoxes(name='boxes')([anchors_input, regression[..., :4]])  # type:ignore
    boxes = ClipBoxes(name='clipped_boxes')([image_input, boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    if detect_quadrangle:
        detections = FilterDetections(
            name='filtered_detections',
            score_threshold=score_threshold,
            detect_quadrangle=True
        )([boxes, classification, regression[..., 4:8], regression[..., 8]])  # type: ignore
    else:
        detections = FilterDetections(
            name='filtered_detections',
            score_threshold=score_threshold
        )([boxes, classification])

    prediction_model = models.Model(inputs=[image_input], outputs=detections, name='efficientdet_p')
    return prediction_model


if __name__ == '__main__':
    x = efficientdet(1)
