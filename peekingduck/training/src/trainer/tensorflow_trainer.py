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
from sklearn.metrics import log_loss
from hydra.utils import instantiate
from omegaconf import DictConfig
import tensorflow as tf
import logging

from src.trainer.base import Trainer
from src.optimizers.adapter import OptimizersAdapter
from src.optimizers.schedules import OptimizerSchedules
from src.model.tensorflow_model import TFClassificationModelFactory
from src.losses.adapter import TensorFlowLossAdapter
from src.metrics.tensorflow_metrics import TensorflowMetrics
from src.callbacks.tensorflow_callbacks import TensorFlowCallbacksAdapter
from src.utils.general_utils import merge_dict_of_list
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

        # create model
        self.model = TFClassificationModelFactory.create_model(
            model_config[self.framework]
        )

        # scheduler
        if self.trainer_config.lr_schedule_params.schedule is None:
            self.scheduler = (
                self.trainer_config.lr_schedule_params.schedule_params.learning_rate
            )
        else:
            self.scheduler = OptimizerSchedules.get_scheduler(
                self.trainer_config.lr_schedule_params.schedule,
                self.trainer_config.lr_schedule_params.schedule_params,
            )

        # init_optimizer
        self.opt = OptimizersAdapter.get_optimizer(
            self.trainer_config.optimizer_params.optimizer,
            self.scheduler,
            self.trainer_config.optimizer_params.optimizer_params,
        )

        # loss
        self.loss = TensorFlowLossAdapter.get_loss_func(
            self.trainer_config.loss_params.loss_func,
            self.trainer_config.loss_params.loss_params,
        )

        # metric
        self.metrics: List = TensorflowMetrics().get_metrics(
            metrics=metrics_config[self.framework]
        )

        # callback
        self.callbacks: List = TensorFlowCallbacksAdapter().get_callbacks(
            callbacks=callbacks_config[self.framework]
        )

        # compile model
        self.model.compile(optimizer=self.opt, loss=self.loss, metrics=self.metrics)

    def train(self, train_dl, val_dl):
        self.model.summary()

        self.epochs = self.trainer_config.global_train_params.epochs
        if self.trainer_config.global_train_params.debug:
            self.epochs = self.trainer_config.global_train_params.debug_epochs

        feature_extraction_history = self.model.fit(
            train_dl,
            epochs=self.epochs,
            validation_data=val_dl,
            callbacks=self.callbacks,
        )
        # Unfreeze the base model
        self.model.trainable = True
        # print(self.model_config)
        for layer in self.model.get_layer("vgg16").layers[:-4]:
            layer.trainable = False
            print(layer.name)
        self.model.summary()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss=self.loss,
            metrics=self.metrics,
        )
        fine_tuning_history = self.model.fit(
            train_dl,
            epochs=self.epochs,
            validation_data=val_dl,
            callbacks=self.callbacks,
        )
        history = merge_dict_of_list(feature_extraction_history.history, fine_tuning_history.history)
        return history
