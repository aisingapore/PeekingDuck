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

import logging

import tensorflow as tf
from omegaconf import DictConfig

# from src.model.tensorflow_base import TFModel
from src.model.tensorflow_base import TFModelFactory  # for testing


logger = logging.getLogger("TF_model")  # pylint: disable=invalid-name
logging.basicConfig(level=logging.INFO)


class TFClassificationModelFactory(TFModelFactory):
    """Generic TensorFlow image classification model."""

    def __init__(self) -> None:
        self.model_cfg = None
        self.model_name = None
        self.input_shape = None
        self.num_classes = None
        # self.model = self.create_model()
        # logger.info("model built!")

    # def data_processing(self, inputs):
    #     data_augmentation = tf.keras.Sequential(
    #         [
    #             tf.keras.layers.RandomFlip("horizontal"),
    #             tf.keras.layers.RandomRotation(0.2),
    #         ]
    #     )
    #     preprocess_input = getattr(
    #         tf.keras.applications, "mobilenet_v2"
    #     ).preprocess_input
    #     x = data_augmentation(inputs)
    #     outputs = preprocess_input(x)
    #     return outputs

    def create_base(self):
        base_model = getattr(tf.keras.applications, self.model_name)(
            input_shape=self.input_shape, include_top=False, weights="imagenet"
        )
        # To-do: allow user to unfreeze certain number of layers for fine-tuning
        base_model.trainable = False
        return base_model

    def create_head(self, inputs):
        # create the pooling and prediction layers
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(self.num_classes, activation="softmax")
        # chain up the layers
        x = global_average_layer(inputs)
        outputs = prediction_layer(x)
        return outputs

    def create_model(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg
        self.model_name = self.model_cfg.model_name
        # print("IMAGE SIZE: ", int(self.model_cfg.image_size))
        self.input_shape = (
            int(self.model_cfg.image_size),
            int(self.model_cfg.image_size),
            3,
        )
        self.num_classes = self.model_config.num_classes
        inputs = tf.keras.Input(shape=self.input_shape)
        # x = self.data_processing(inputs)
        # disable batch norm for base model
        x = self.create_base()(inputs, training=False)
        outputs = self.create_head(x)
        model = tf.keras.Model(inputs, outputs)
        logger.info("model created!")
        return model
