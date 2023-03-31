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

""" Training a neural network on MNIST with Keras
This simple example demonstrates how to plug TensorFlow Datasets (TFDS) into a Keras model.
"""

from typing import Any, Tuple
from pytest import mark

import tensorflow as tf
import tensorflow_datasets as tfds


@mark.skip(reason="Redundant tensorflow test")
def test_tensorflow_data_module() -> None:
    """Test Tensorflow datasource"""
    # Step 1: Create your input pipeline
    # ### Load a dataset
    (ds_train, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # ### Build a training pipeline
    def normalize_img(image: tf.Tensor, label: tf.Tensor) -> Tuple[Any, Any]:
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255.0, label

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # ### Build an evaluation pipeline
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    # ## Step 2: Create and train the model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="gelu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    history = model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
    )

    assert history is not None
    assert hasattr(history, "history")
    assert "val_sparse_categorical_accuracy" in history.history
    assert len(history.history["val_sparse_categorical_accuracy"]) != 0
    assert history.history["val_sparse_categorical_accuracy"][-1] >= 0.97


if __name__ == "__main__":
    test_tensorflow_data_module()
