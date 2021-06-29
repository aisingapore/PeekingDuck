# Copyright 2021 AI Singapore
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
Make trainable=False freeze BN
"""

import numpy as np
import tensorflow as tf


class BatchNormalization(tf.keras.layers.BatchNormalization):  # pylint: disable=too-many-ancestors, too-few-public-methods
    """
    Make trainable=False freeze BN for real (the og version is sad)
    """

    def call(self, inputs: np.array, training: tf.Tensor = None) -> object:
        """Make trainable=False freeze BN for real (the og version is sad)

        Input:
            - inputs: input image matrix
            - training: the training data

        Output:
            - inputs_image: transformed image matrix
        """
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(inputs, training)
