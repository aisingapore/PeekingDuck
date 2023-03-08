# Copyright 2023 AI Singapore
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

from typing import List

import tensorflow as tf


def unfreeze_all_layers(model: tf.keras.Model) -> None:
    for layer in model.layers:
        layer.trainable = True


def freeze_all_layers(model: tf.keras.Model) -> None:
    for layer in model.layers:
        layer.trainable = False


def set_trainable_layers(
    model: tf.keras.Model, trainable_layer_name_list: List[str]
) -> None:
    """Set the layers in the model to be trainable.
    If a layer is not included in the list, it will be freezed.
    """
    # Needed because if model.trainable = False, the layer.trainable = True will not take effect.
    # Reference: https://github.com/tensorflow/tensorflow/issues/29535
    # model.trainable = True # comment out for testing first. Remove later if not needed,

    # freeze the layers that are not in the provided layer list
    for layer in model.layers:
        if layer.name not in trainable_layer_name_list:
            layer.trainable = False
        else:
            layer.trainable = True
