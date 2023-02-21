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
from src.losses.adapter import LossAdapter
from src.metrics.tensorflow_metrics import TensorflowMetrics
from src.callbacks.tensorflow_callbacks import TensorFlowCallbacksAdapter
from src.utils.general_utils import merge_dict_of_list
from src.utils.tf_model_utils import set_trainable_layers
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
        device: str,
    ) -> None:
        """Called when the trainer begins."""
        self.trainer_config = trainer_config[self.framework]
        self.model_config = model_config[self.framework]
        self.metrics_config = metrics_config[self.framework]
        self.callbacks_config = callbacks_config[self.framework]

        # create model
        self.model = TFClassificationModelFactory.create_model(self.model_config)

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
        self.opt = OptimizersAdapter.get_tensorflow_optimizer(
            self.trainer_config.optimizer_params.optimizer,
            self.scheduler,
            self.trainer_config.optimizer_params.optimizer_params,
        )

        # loss
        self.loss = LossAdapter.get_tensorflow_loss_func(
            self.trainer_config.loss_params.loss_func,
            self.trainer_config.loss_params.loss_params,
        )

        # metric
        self.metrics: List = TensorflowMetrics().get_metrics(
            metrics=self.metrics_config
        )

        # callback
        self.callbacks: List = TensorFlowCallbacksAdapter().get_callbacks(
            callbacks=self.callbacks_config
        )

        # compile model
        self.model.compile(optimizer=self.opt, loss=self.loss, metrics=self.metrics)

    def train(self, train_dl, val_dl):
        self.model.summary(expand_nested=True)

        # test setting
        set_trainable_layers(self.model, self.model_config.fine_tune_layers)
        self.model.summary(expand_nested=True)

        self.epochs = self.trainer_config.global_train_params.epochs
        if self.trainer_config.global_train_params.debug:
            self.epochs = self.trainer_config.global_train_params.debug_epochs

        feature_extraction_history = self.model.fit(
            train_dl,
            epochs=self.epochs,
            validation_data=val_dl,
            callbacks=self.callbacks,
        )

        # if self.model_config.unfreeze_layers <= 0:
        #     return feature_extraction_history.history

        # # Unfreeze the base model
        # self.model.trainable = True
        # # cannot set to False, otherwise the layer.trainable attribute will not work.

        # if self.model_config.unfreeze_layers > 0:
        #     # get into the layers within the backbone and unfreeze
        #     for layer in self.model.get_layer(
        #         str(self.model_config.model_name).lower()
        #     ).layers[: -abs(self.model_config.unfreeze_layers)]:
        #         layer.trainable = False  # Freeze layers up to the number

        # return the current history if no fine_tune required. Continue training otherwise
        if self.model_config.fine_tune == False:
            return feature_extraction_history.history
        elif self.model_config.fine_tune == True:

            set_trainable_layers(self.model, self.model_config.fine_tune_layers)

            self.model.summary()

            optimizer = OptimizersAdapter.get_tensorflow_optimizer(
                self.trainer_config.optimizer_params.optimizer,
                self.trainer_config.fine_tune_params.optimizer_learning_rate,
                self.trainer_config.optimizer_params.optimizer_params,
            )

            self.model.compile(
                optimizer=optimizer,
                loss=self.loss,
                metrics=self.metrics,
            )

            fine_tuning_history = self.model.fit(
                train_dl,
                epochs=self.epochs,
                validation_data=val_dl,
                callbacks=self.callbacks,
            )
            history = merge_dict_of_list(
                feature_extraction_history.history, fine_tuning_history.history
            )
        else:
            logger.error(f"Unknown fine_tune setting '{self.model_config.fine_tune}'")
        return history
