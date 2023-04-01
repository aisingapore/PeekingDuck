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

"""tensorflow models"""

import logging

import tensorflow as tf
from omegaconf import DictConfig

from src.model.tensorflow_base import TFModelFactory
from src.utils.tf_model_utils import set_trainable_layers

# pylint: disable=invalid-name,too-few-public-methods
logger = logging.getLogger("TF Model")
logging.basicConfig(level=logging.INFO)


class TFClassificationModelFactory(TFModelFactory):
    """Generic TensorFlow image classification model."""

    @classmethod
    def create_model(cls, model_config: DictConfig) -> tf.keras.Model:
        """model factory method"""
        model_name = model_config.model_name
        num_classes = model_config.num_classes
        prediction_layer_name = "prediction_modified"

        input_tensor = tf.keras.layers.Input(
            shape=(model_config.image_size, model_config.image_size, 3)
        )
        model: tf.keras.Model = getattr(tf.keras.applications, model_name)(
            input_tensor=input_tensor, include_top=True, weights="imagenet"
        )

        # exclude the existing prediction layer
        x = model.layers[-2].output
        # create the new prediction layer
        predictions = tf.keras.layers.Dense(
            num_classes, activation=model_config.activation, name=prediction_layer_name
        )(x)
        # Create new model with modified classification layer
        model = tf.keras.Model(inputs=model.inputs, outputs=predictions)
        # freeze all the layers except the prediction layer
        set_trainable_layers(model, [prediction_layer_name])

        logger.info("model created!")
        return model
