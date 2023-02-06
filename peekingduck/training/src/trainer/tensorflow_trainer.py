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

from sklearn.metrics import log_loss
from src.trainer.base import Trainer
from hydra.utils import instantiate
from omegaconf import DictConfig

import tensorflow as tf
import logging
from configs import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name


class tensorflowTrainer(Trainer):
    def __init__(self, framework: str = "tensorflow") -> None:
        self.framework = framework
        self.model = None
        self.scheduler = None
        self.opt = None
        self.loss = None
        self.metrics = None
        self.callbacks = None

    def setup(
        self,
        trainer_config: DictConfig,
        model_config: DictConfig,
        callbacks_config: DictConfig,
        metrics_config: DictConfig,
        data_config: DictConfig,
        device: str = "cpu",
    ) -> None:
        """Called when the trainer begins."""
        self.trainer_config = trainer_config[self.framework]

        tf_model_factory = instantiate(model_config[self.framework].model_type)
        self.model = tf_model_factory.create_model(model_config[self.framework])
        # self.model = tf.keras.applications.ResNet50(input_shape=(32, 32, 3), classes=10, include_top=True, weights=None)

        # scheduler
        decay_steps = 1000
        initial_learning_rate = 0.001
        self.scheduler = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate, decay_steps
        )

        # init_optimizer
        # adam = tf.keras.optimizers.Adam(
        adam = tf.keras.optimizers.legacy.Adam(
            learning_rate=self.scheduler,
            beta_1=0.9,
            beta_2=0.999,
            name="Adam",
        )
        self.opt = adam

        # loss
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        self.loss = cce

        # metric
        metrics_adapter = instantiate(metrics_config[self.framework].adapter)
        metrics_adapter.setup(
            task=data_config.dataset.classification_type,
            num_classes=data_config.dataset.num_classes,
            metrics=metrics_config[self.framework].evaluate,
        )
        self.metrics = metrics_adapter.create_collection()

        # callback
        es = tf.keras.callbacks.EarlyStopping(
            patience=3, restore_best_weights=True, monitor="accuracy"
        )
        self.callbacks = [es]

        # self.model.compile(optimizer=self.opt, loss=self.loss, metrics=self.metrics)
        self.model.compile(optimizer = self.opt,loss=self.loss, metrics=["accuracy"]) 

    def train(self, train_dl, val_dl):
        BATCH_SIZE = 32
        EPOCHS = 10

        self.model.summary()
        history = self.model.fit(
            train_dl,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=val_dl,
            callbacks=self.callbacks,
        )

        return history
