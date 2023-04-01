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

""" Test data module, data adapter, data loader and dataset
"""

from typing import List
from pytest import mark

from hydra import compose, initialize
import torch
import tensorflow as tf

from src.data.data_module import ImageClassificationDataModule
from src.model_analysis.weights_biases import WeightsAndBiases
from src.training_pipeline import init_trainer


@mark.parametrize(
    "overrides, expected",
    [
        (
            [
                "project_name=cifar10",
                "data_module=cifar10",
                "framework=tensorflow",
                "debug=True",
                "device=cpu",
                "data_module.dataset.image_size=224",
                "data_module.dataset.download=True",
                "data_module.data_adapter.tensorflow.train.batch_size=32",
                "data_module.data_adapter.tensorflow.validation.batch_size=32",
                "data_module.data_adapter.tensorflow.test.batch_size=32",
            ],
            (32, 224, 224, 3),
        ),
        (
            [
                "project_name=cifar10",
                "data_module=cifar10",
                "framework=pytorch",
                "debug=True",
                "device=cpu",
                "data_module.dataset.image_size=32",
                "data_module.dataset.download=True",
                "data_module.data_adapter.pytorch.train.batch_size=32",
                "data_module.data_adapter.pytorch.validation.batch_size=32",
                "data_module.data_adapter.pytorch.test.batch_size=32",
                "trainer.pytorch.stores.model_artifacts_dir=null",
            ],
            torch.Size([32, 3, 32, 32]),
        ),
    ],
)
def test_data_module(overrides: List[str], expected: List[int]) -> None:
    """Test data_module"""
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(
            config_name="config",
            overrides=overrides,
        )

        data_module = ImageClassificationDataModule(
            cfg=cfg.data_module,
        )

        data_module.prepare_data()
        data_module.setup(stage="fit")
        train_loader = data_module.get_train_dataloader()
        validation_loader = data_module.get_validation_dataloader()
        assert train_loader
        inputs, _ = next(iter(train_loader))
        assert inputs.shape == expected

        assert validation_loader
        inputs, _ = next(iter(validation_loader))
        assert inputs.shape == expected


@mark.parametrize(
    "overrides, validation_loss_key, expected",
    [
        (
            [
                "project_name=cifar10",
                "data_module=cifar10",
                "framework=tensorflow",
                "debug=True",
                "device=cpu",
                "data_module.dataset.image_size=32",
                "data_module.dataset.download=True",
                "data_module.data_adapter.tensorflow.train.batch_size=32",
                "data_module.data_adapter.tensorflow.validation.batch_size=32",
                "data_module.data_adapter.tensorflow.test.batch_size=32",
            ],
            "val_loss",
            3.0,
        ),
    ],
)
def test_tensorflow_trainer(
    overrides: List[str], validation_loss_key: str, expected: float
) -> None:
    """Test data_module"""
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(
            config_name="config",
            overrides=overrides,
        )

        data_module = ImageClassificationDataModule(
            cfg=cfg.data_module,
        )

        data_module.prepare_data()
        data_module.setup(stage="fit")
        train_loader = data_module.get_train_dataloader()
        validation_loader = data_module.get_validation_dataloader()

        num_classes = 10
        input_shape = (32, 32, 3)

        history = None
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=input_shape),
                tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )

        ## Train the model
        epochs = 5

        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        history = model.fit(
            train_loader, epochs=epochs, validation_data=validation_loader
        )
        assert history is not None
        assert hasattr(history, "history")
        assert validation_loss_key in history.history
        assert len(history.history[validation_loss_key]) != 0
        assert history.history[validation_loss_key][-1] < expected


@mark.parametrize(
    "overrides, validation_loss_key, expected",
    [
        (
            [
                "project_name=cifar10",
                "data_module=cifar10",
                "framework=pytorch",
                "debug=True",
                "device=cpu",
                "data_module.dataset.image_size=32",
                "data_module.dataset.download=True",
                "data_module.data_adapter.pytorch.train.batch_size=32",
                "data_module.data_adapter.pytorch.validation.batch_size=32",
                "data_module.data_adapter.pytorch.test.batch_size=32",
                "trainer.pytorch.stores.model_artifacts_dir=null",
            ],
            "valid_loss",
            2.3,
        ),
    ],
)
def test_pytorch_trainer(
    overrides: List[str], validation_loss_key: str, expected: float
) -> None:
    """Test pytorch data_module"""
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(
            config_name="config",
            overrides=overrides,
        )

        data_module = ImageClassificationDataModule(
            cfg=cfg.data_module,
        )

        data_module.prepare_data()
        data_module.setup(stage="fit")
        train_loader = data_module.get_train_dataloader()
        validation_loader = data_module.get_validation_dataloader()

        history = None
        WeightsAndBiases(cfg.model_analysis)
        trainer = init_trainer(cfg)
        history = trainer.train(train_loader, validation_loader)
        assert history is not None
        assert validation_loss_key in history
        assert len(history[validation_loss_key]) != 0
        assert history[validation_loss_key][-1] < expected
