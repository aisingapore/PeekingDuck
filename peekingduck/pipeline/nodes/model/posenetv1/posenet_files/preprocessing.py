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
from typing import List

import cv2
import numpy as np

from peekingduck.pipeline.nodes.model.posenetv1.posenet_files.constants import IMAGE_NET_MEAN


def rescale_image(source_img: List[List[float]],
                  input_res: List[int],
                  scale_factor: int = 1.0,
                  output_stride: int = 16,
                  model_type: str = 'mobilenet'):
    """Rescale the image by a scale factor while ensuring it has a valid output
    stride

    Args:
        source_img (np.array): image for inference
        scale_factor (int): ratio to scale image
        output_stride (int): output stride to convert output indices to image coordinates

    Returns:
        image_processed (np.array): processed image
        scale (np.array): factor to scale height and width
    """
    target_width, target_height = _get_valid_resolution(
        input_res[0] * scale_factor,
        input_res[1] * scale_factor,
        output_stride=output_stride)

    scale = np.array([
        source_img.shape[1] / target_width,
        source_img.shape[0] / target_height
    ])

    image_processed = _rescale_image(source_img, target_width, target_height,
                                     model_type)

    return image_processed, scale


def _get_valid_resolution(width: int,
                          height: int,
                          output_stride: int = 16):
    """

    Args:
        width (int): input width
        width (int): input height
        output_stride (int): output stride to convert output indices to image coordinates

    Returns:
        target_width (int): output width
        target_width (int): output height
    """

    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height


def _rescale_image(source_img: List[List[float]],
                   target_width: int,
                   target_height: int,
                   model_type: str):
    """ Apply different preprocessing according to model type - mobilenet or resnet

    For mobilenet version, RGB values were scaled (2.0 / 255.0) - 1.0
    For resnet version, IMAGE_NET_MEAN were added to the RGB values

    Args:
        source_img (np.array): image for inference
        target_width (int): output width
        target_width (int): output height
        model_type (str): specified model type (refer to modelconfig.yml)

    Returns:
        image_processed (np.array): proccessed image array
    """
    image_processed = cv2.resize(source_img, (target_width, target_height),
                                 interpolation=cv2.INTER_LINEAR)
    image_processed = cv2.cvtColor(image_processed,
                                   cv2.COLOR_BGR2RGB).astype(np.float32)

    if model_type == 'resnet':
        image_processed += IMAGE_NET_MEAN
    else:
        image_processed = image_processed * (2.0 / 255.0) - 1.0

    image_processed = image_processed.reshape(1,
                                              target_height,
                                              target_width,
                                              3)
    return image_processed
