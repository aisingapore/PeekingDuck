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

import os
import logging

# import matplotlib.pyplot as plt
import tensorflow as tf

from src.model.tensorflow_base import Model

logger = logging.getLogger("TF_test")  # pylint: disable=invalid-name
logging.basicConfig(level=logging.INFO)


class ImageClassificationModel(Model):
    """Generic TensorFlow image classification model."""

    def __init__(self) -> None:
        super().__init__()
        self.model_name = "MobileNetV2"
        self.input_shape = (160, 160, 3)
        self.learning_rate = 0.0001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.metrics = ["accuracy"]
        self.__init_data()
        self.create_model()
        logger.info("model created!")

    def __init_data(self) -> None:
        # download dataset
        _URL = (
            "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
        )
        path_to_zip = tf.keras.utils.get_file(
            "cats_and_dogs.zip", origin=_URL, extract=True
        )
        PATH = os.path.join(os.path.dirname(path_to_zip), "cats_and_dogs_filtered")

        train_dir = os.path.join(PATH, "train")
        validation_dir = os.path.join(PATH, "validation")

        BATCH_SIZE = 32
        IMG_SIZE = (160, 160)

        # assign train and validation datasets to the instance
        self.train_dataset = tf.keras.utils.image_dataset_from_directory(
            train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
        )
        self.validation_dataset = tf.keras.utils.image_dataset_from_directory(
            validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
        )

    def data_processing(self, inputs):
        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.2),
            ]
        )

        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        x = data_augmentation(inputs)
        outputs = preprocess_input(x)
        return outputs

    def create_base(self):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape, include_top=False, weights="imagenet"
        )

        # freeze the base
        # To-do: allow user to unfreeze certain number of layers for fine-tuning
        base_model.trainable = False
        return base_model

    def create_head(self, inputs):
        # create the pooling and prediction layers
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(1)
        # chain up the layers
        x = global_average_layer(inputs)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        return outputs

    def build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = self.data_processing(inputs)
        x = self.create_base()(x, training=False)  # disable batch norm for base model
        outputs = self.create_head(x)
        self.model = tf.keras.Model(inputs, outputs)
        logger.info("model built!")

    def compile_model(self):
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
        )
        logger.info("model compiled!")
        self.model.summary()

    def create_model(self):
        self.build_model()
        self.compile_model()


"""
classification_model = ImageClassificationModel()

# prepare for training
initial_epochs = 5
loss0, accuracy0 = classification_model.model.evaluate(
    classification_model.validation_dataset
)
logger.info("initial loss: {:.2f}".format(loss0))
logger.info("initial accuracy: {:.2f}".format(accuracy0))

# training
history = classification_model.model.fit(
    classification_model.train_dataset,
    epochs=initial_epochs,
    validation_data=classification_model.validation_dataset,
)

# show leaning curve
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.ylabel("Accuracy")
plt.ylim([min(plt.ylim()), 1])
plt.title("Training and Validation Accuracy")

plt.subplot(2, 1, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.ylabel("Cross Entropy")
plt.ylim([0, 1.0])
plt.title("Training and Validation Loss")
plt.xlabel("epoch")
plt.show()
"""
