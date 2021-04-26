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

import cv2
import numpy as np

IMAGE_NET_MEAN = [-123.15, -115.90, -103.06]


def rescale_image(source_img, input_res, scale_factor=1.0, output_stride=16,
                  model_type='mobilenet'):
    """
    rescale the image by a scale factor while ensuring it has a valid output
    stride

    args:
        source_img - source np.array image
        scale_factor - scale factor
        output_stride - the stride to ensure

    return:
        the new scaled image and the effective scale in width/height
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


def _get_valid_resolution(width, height, output_stride=16):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height


def _rescale_image(source_img, target_width, target_height, model_type):
    '''
    Rescales the image using the appropriate process. Since posenet was trained
    by the tensorflow team, they used different preprocessing for the mobilenet
    and resnet version.

    For mobilenet version, the preprocessing was a simple (2.0 / 255.0) - 1.0

    For resnet version, it was adding the IMAGE_NET_MEAN, which is defined at
    the top of this script.

    You can find the implementation here:
    https://github.com/tensorflow/tfjs-models/blob/master/posenet/src/resnet.ts

    This is why the check for resnet is needed below.
    '''
    image_processed = cv2.resize(source_img, (target_width, target_height),
                                 interpolation=cv2.INTER_LINEAR)
    image_processed = cv2.cvtColor(image_processed,
                                   cv2.COLOR_BGR2RGB).astype(np.float32)

    # See docstring for reason why this is needed
    if model_type == 'resnet':
        image_processed += IMAGE_NET_MEAN
    else:
        image_processed = image_processed * (2.0 / 255.0) - 1.0

    image_processed = image_processed.reshape(1,
                                              target_height,
                                              target_width,
                                              3)
    return image_processed
