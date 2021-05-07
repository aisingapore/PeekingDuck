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

import tensorflow as tf


def transform_images(x_image: tf.Tensor, size: int) -> tf.Tensor:
    """transform image to size x size

    Input:
        - x_image: input image matrix
        - size: integer size of the image

    Output:
        - x_image: transformed image matrix
    """
    x_image = tf.image.resize(x_image, (size, size))
    x_image = x_image / 255
    return x_image
