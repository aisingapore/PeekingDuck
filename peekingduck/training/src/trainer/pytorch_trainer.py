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

import numpy as np
import pandas as pd
import time
import torch

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Union
from tabulate import tabulate

from tqdm.auto import tqdm
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from hydra.utils import instantiate

from src.optimizers.schedules import OptimizerSchedules
from src.losses.adapter import LossAdapter
from src.optimizers.adapter import OptimizersAdapter
from src.trainer.base import Trainer
from src.callbacks.base import init_callbacks
from src.callbacks.events import EVENTS
from src.model.pytorch_base import PTModel
from src.metrics.pytorch_metrics import PytorchMetrics
from src.utils.general_utils import free_gpu_memory  # , init_logger


# TODO: clean up val vs valid naming confusions.
def get_sigmoid_softmax(
    trainer_config: DictConfig,
) -> Union[torch.nn.Sigmoid, torch.nn.Softmax]:
    """Get the sigmoid or softmax function depending on loss function."""
    if trainer_config.criterion_params.train_criterion == "BCEWithLogitsLoss":
        return getattr(torch.nn, "Sigmoid")()

    if trainer_config.criterion_params.train_criterion == "CrossEntropyLoss":
        return getattr(torch.nn, "Softmax")(dim=1)


class pytorchTrainer(Trainer):
    """Object used to facilitate training."""

    def __init__(self, framework: str = "pytorch") -> None:
        """Initialize the trainer."""
        self.framework = framework
        self.device = None

        self.trainer_config = None

        self.callbacks = None
        self.metrics = None
        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.train_params = None
        self.model_artifacts_dir = None
        self.monitored_metric = None
        self.best_val_score = None

        self.train_loader = None
        self.validation_loader = None

        self.stop_training = False
        self.history = defaultdict(list)
        self.epochs = None
        self.current_epoch = None
        self.epoch_dict = None
        self.batch_dict = None
        self.history_dict = None
        self.metrics_dict = None

        self.current_fold = None
        self.train_start_time = None
        self.train_time_elapsed = None
        self.valid_elapsed_time = None
        self.valid_elapsed_time = None

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
        self.model_config = model_config[self.framework]
        self.callbacks_config = callbacks_config[self.framework]
        self.metrics_config = metrics_config[self.framework]
        self.train_params = self.trainer_config.global_train_params
        self.model_artifacts_dir = self.trainer_config.stores.model_artifacts_dir
        self.device = device
        self.callbacks = init_callbacks(callbacks_config[self.framework])
        self.metrics = PytorchMetrics.get_metrics(
            task=data_config.dataset.classification_type,
            num_classes=data_config.dataset.num_classes,
            metric_list=metrics_config[self.framework],
        )

        # create model
        self.model: PTModel = instantiate(
            config=self.model_config.model_type,
            cfg=self.model_config,
            _recursive_=False,
        ).to(self.device)

        # init_optimizer
        self.optimizer = OptimizersAdapter.get_pytorch_optimizer(
            model=self.model,
            optimizer_params=self.trainer_config.optimizer_params,
        )

        # scheduler
        if not self.trainer_config.scheduler_params.scheduler is None:
            self.scheduler = OptimizerSchedules.get_pytorch_scheduler(
                optimizer=self.optimizer,
                scheduler=self.trainer_config.scheduler_params.scheduler,
                parameters=self.trainer_config.scheduler_params.scheduler_params,
            )

        self.monitored_metric = self.train_params.monitored_metric

        # Metric to optimize, either min or max.
        self.best_val_score = (
            -np.inf if self.monitored_metric["mode"] == "max" else np.inf
        )

        self.current_epoch = 1
        self.epoch_dict = {}
        self.epoch_dict["train"] = {}
        self.epoch_dict["validation"] = {}

        self.batch_dict = {}

        self.history_dict = {}
        self.history_dict["train"] = {}
        self.history_dict["validation"] = {}

        self.metrics_dict = {}
        self.metrics_dict["train"] = {}
        self.metrics_dict["validation"] = {}
        self._invoke_callbacks(EVENTS.ON_TRAINER_START.value)

    def _set_dataloaders(
        self,
        train_dl: DataLoader,
        validation_dl: DataLoader,
    ) -> None:
        """Initialise Dataloader Variables"""
        self.train_loader = train_dl
        self.validation_loader = validation_dl

    def _train_setup(self, inputs) -> None:
        # show model summary
        self.model.model_summary(inputs.shape)
        self.best_valid_loss = np.inf

    def _train_teardown(self) -> None:
        free_gpu_memory(
            self.optimizer,
            self.scheduler,
            self.history_dict["validation"]["valid_trues"],
            self.history_dict["validation"]["valid_logits"],
            self.history_dict["validation"]["valid_preds"],
            self.history_dict["validation"]["valid_probs"],
        )

    def _run_epochs(self) -> None:
        self.epochs = self.train_params.epochs
        if self.train_params.debug:
            self.epochs = self.train_params.debug_epochs

        # implement
        for epoch in range(1, self.epochs + 1):
            self.curr_epoch = epoch
            self._run_train_epoch(self.train_loader)
            self._run_validation_epoch(self.validation_loader)

            if self.stop_training:  # from early stopping
                break  # Early Stopping

            if self.scheduler is not None:
                # Special Case for ReduceLROnPlateau
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(self.monitored_metric["metric_score"])
                else:
                    self.scheduler.step()

            self.epoch_dict["train"]["epoch"] = self.curr_epoch
            self.epoch_dict["validation"]["epoch"] = self.curr_epoch
            self.current_epoch += 1

    def _run_train_epoch(self, train_loader: DataLoader) -> None:
        """Train one epoch of the model."""
        self.train_start_time = time.time()
        self._invoke_callbacks(EVENTS.ON_TRAIN_EPOCH_START.value)

        self.curr_lr = LossAdapter.get_lr(self.optimizer)
        # set to train mode
        self.model.train()

        train_bar = tqdm(train_loader)
        train_trues, train_logits, train_preds, train_probs = [], [], [], []

        self._invoke_callbacks(EVENTS.ON_TRAIN_LOADER_START.value)
        # Iterate over train batches
        for _, batch in enumerate(train_bar, start=1):
            self._invoke_callbacks(EVENTS.ON_TRAIN_BATCH_START.value)

            # unpack - note that if BCEWithLogitsLoss, dataset should do view(-1,1) and not here.
            inputs, targets = batch
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(inputs)  # Forward pass logits
            curr_batch_train_loss = LossAdapter.compute_criterion(
                targets,
                logits,
                criterion_params=self.trainer_config.criterion_params,
                stage="train",
            )

            self.optimizer.zero_grad()  # reset gradients
            curr_batch_train_loss.backward()  # Backward pass
            self.optimizer.step()  # Update weights using the optimizer

            # Update loss metric, every batch is diff
            self.batch_dict["train_loss"] = curr_batch_train_loss.item()
            y_train_prob = get_sigmoid_softmax(self.trainer_config)(logits)
            y_train_pred = torch.argmax(y_train_prob, dim=1)

            self._invoke_callbacks(EVENTS.ON_TRAIN_BATCH_END.value)
            
            train_trues.extend(targets.cpu())
            train_logits.extend(logits.cpu())
            train_probs.extend(y_train_prob.cpu())
            train_preds.extend(y_train_pred.cpu())

        train_trues, train_logits,  train_preds,  train_probs = (
            torch.vstack(train_trues),
            torch.vstack(train_logits),
            torch.vstack(train_preds),
            torch.vstack(train_probs),
        )

        _, self.metrics_dict["train"]["train_metrics_df"] = PytorchMetrics.get_classification_metrics(
            self.metrics,
            train_trues,
            train_preds,
            train_probs,
            "train",
        )

        self._invoke_callbacks(EVENTS.ON_TRAIN_LOADER_END.value)

        self.history_dict["train"] = {**self.epoch_dict["train"]}
        # total time elapsed for this epoch
        self.train_time_elapsed = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - self.train_start_time)
        )
        self._invoke_callbacks(EVENTS.ON_TRAIN_EPOCH_END.value)

    def _run_validation_epoch(self, validation_loader: DataLoader) -> None:
        """Validate the model on the validation set for one epoch.
        Args:
            validation_loader (torch.utils.data.DataLoader): The validation set dataloader.
        Returns:
            Dict[str, np.ndarray]:
                valid_loss (float): The validation loss for each epoch.
                valid_trues (np.ndarray): The ground truth labels for each validation set. shape = (num_samples, 1)
                valid_logits (np.ndarray): The logits for each validation set. shape = (num_samples, num_classes)
                valid_preds (np.ndarray): The predicted labels for each validation set. shape = (num_samples, 1)
                valid_probs (np.ndarray): The predicted probabilities for each validation set. shape = (num_samples, num_classes)
        """
        val_start_time = time.time()  # start time for validation
        self._invoke_callbacks(EVENTS.ON_VALID_EPOCH_START.value)

        self.model.eval()  # set to eval mode

        valid_bar = tqdm(validation_loader)

        valid_trues, valid_logits, valid_preds, valid_probs = [], [], [], []

        self._invoke_callbacks(EVENTS.ON_VALID_LOADER_START.value)

        with torch.no_grad():  # TODO
            for _step, batch in enumerate(valid_bar, start=1):
                self._invoke_callbacks(EVENTS.ON_VALID_BATCH_START.value)

                # unpack
                inputs, targets = batch
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()  # reset gradients

                logits = self.model(inputs)  # Forward pass logits

                # get batch size, may not be same as params.batch_size due to whether drop_last in loader is True or False.
                _batch_size = inputs.shape[0]

                # TODO: Refer to my RANZCR notes on difference between Softmax and Sigmoid with examples.
                y_valid_prob = get_sigmoid_softmax(self.trainer_config)(logits)
                y_valid_pred = torch.argmax(y_valid_prob, axis=1)

                curr_batch_val_loss = LossAdapter.compute_criterion(
                    targets,
                    logits,
                    criterion_params=self.trainer_config.criterion_params,
                    stage="valid",
                )
                # Update loss metric, every batch is diff
                self.batch_dict["valid_loss"] = curr_batch_val_loss.item()

                # valid_bar.set_description(f"Validation. {metric_monitor}")

                self._invoke_callbacks(EVENTS.ON_VALID_BATCH_END.value)
                # For OOF score and other computation.
                # TODO: Consider giving numerical example. Consider rolling back to targets.cpu().numpy() if torch fails.
                valid_trues.extend(targets.cpu())
                valid_logits.extend(logits.cpu())
                valid_preds.extend(y_valid_pred.cpu())
                valid_probs.extend(y_valid_prob.cpu())

        valid_trues, valid_logits, valid_preds, valid_probs = (
            torch.vstack(valid_trues),
            torch.vstack(valid_logits),
            torch.vstack(valid_preds),
            torch.vstack(valid_probs),
        )
        self.metrics_dict["validation"]["valid_metrics_dict"], self.metrics_dict["validation"]["valid_metrics_df"] = PytorchMetrics.get_classification_metrics(
            self.metrics,
            valid_trues,
            valid_preds,
            valid_probs,
            "val",
        )

        self._invoke_callbacks(EVENTS.ON_VALID_LOADER_END.value)

        # total time elapsed for this epoch
        self.valid_elapsed_time = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - val_start_time)
        )
        self.epoch_dict["validation"].update(
            {
                "valid_trues": valid_trues,
                "valid_logits": valid_logits,
                "valid_preds": valid_preds,
                "valid_probs": valid_probs,
                "valid_elapsed_time": self.valid_elapsed_time,
            }
        )
        # FIXME: potential difficulty in debugging since epoch_dict is called in metrics meter
        self.epoch_dict["validation"].update(self.metrics_dict["validation"]["valid_metrics_dict"])
        self.history_dict["validation"] = {
            **self.epoch_dict["validation"],
            **self.metrics_dict["validation"]["valid_metrics_dict"],
        }
        self._invoke_callbacks(EVENTS.ON_VALID_EPOCH_END.value)

    def _invoke_callbacks(self, event_name: str) -> None:
        """Invoke the callbacks."""
        for callback in self.callbacks:
            try:
                getattr(callback, event_name)(self)
            except NotImplementedError:
                pass

    def train(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        fold: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Fit the model and returns the history object."""
        self.current_fold = fold
        self._invoke_callbacks(EVENTS.ON_TRAINER_START.value)

        self._set_dataloaders(train_dl=train_loader, validation_dl=validation_loader)
        inputs, _ = next(iter(train_loader))
        self._train_setup(inputs)  # startup
        self._run_epochs()
        self._train_teardown()  # shutdown

        self._invoke_callbacks(EVENTS.ON_TRAINER_END.value)
        return self.history
