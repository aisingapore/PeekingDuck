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
import time
import torch

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tqdm.auto import tqdm
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from src.callbacks.default_callbacks import Callback
from src.model.default_model import Model
from src.utils.general_utils import free_gpu_memory  # , init_logger


# TODO: clean up val vs valid naming confusions.
def get_sigmoid_softmax(
    pipeline_config: DictConfig,
) -> Union[torch.nn.Sigmoid, torch.nn.Softmax]:
    """Get the sigmoid or softmax function depending on loss function."""
    if pipeline_config.criterion_params.train_criterion == "BCEWithLogitsLoss":
        return getattr(torch.nn, "Sigmoid")()

    if pipeline_config.criterion_params.train_criterion == "CrossEntropyLoss":
        return getattr(torch.nn, "Softmax")(dim=1)


class Trainer(ABC):
    """Object used to facilitate training."""

    def __init__(
        self,
        pipeline_config: DictConfig,
        model: Model,
        callbacks: List[Callback] = None,
        metrics: Union[MetricCollection, List[str]] = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the trainer."""

        self.pipeline_config = pipeline_config
        self.train_params = self.pipeline_config.global_train_params
        self.model = model
        self.model_artifacts_dir = self.pipeline_config.stores.model_artifacts_dir
        self.device = device

        self.callbacks = callbacks
        self.metrics = metrics

        self.__initialize()  # init non init attributes, etc

        print(self.__dict__)

    def __initialize(self) -> None:
        """Called when the trainer begins."""
        print("TRAINER CLASS optimizer params: ", self.pipeline_config.optimizer_params)
        print("TRAINER CLASS scheduler params: ", self.pipeline_config.scheduler_params)

        self.optimizer = self.get_optimizer(
            model=self.model,
            optimizer_params=self.pipeline_config.optimizer_params,
        )
        self.scheduler = self.get_scheduler(
            optimizer=self.optimizer,
            scheduler_params=self.pipeline_config.scheduler_params,
        )

        self.monitored_metric = self.train_params.monitored_metric

        # Metric to optimize, either min or max.
        self.best_valid_score = (
            -np.inf if self.monitored_metric["mode"] == "max" else np.inf
        )
        self.patience_counter = self.train_params.patience  # Early Stopping Counter
        self.current_epoch = 1
        self.epoch_dict = {}
        self.batch_dict = {}
        self.history_dict = {}
        # self.invoke_callbacks("on_trainer_start")

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        fold: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Fit the model and returns the history object."""
        self.__on_fit_start(fold=fold)

        for _epoch in range(1, self.train_params.epochs + 1):
            self.__train_one_epoch(train_loader, _epoch)
            self.__train_one_epoch(valid_loader, _epoch)

            self.monitored_metric["metric_score"] = torch.clone(
                self.valid_history_dict[self.monitored_metric["monitor"]]
            ).detach()
            valid_loss = self.valid_history_dict["valid_loss"]

            if self.stop:  # from early stopping
                break  # Early Stopping

            if self.scheduler is not None:
                # Special Case for ReduceLROnPlateau
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(self.monitored_metric["metric_score"])
                else:
                    self.scheduler.step()

            self.current_epoch += 1

        self.__on_fit_end()
        # FIXME: here is finish fitting, whether to call it on train end or on fit end?
        # Currently only history uses on_trainer_end.
        for callback in self.callbacks:
            callback.on_trainer_end(self)
        return self.history

    def __on_fit_start(self, fold: int) -> None:
        self.logger.info(f"Fold {fold} started")
        self.best_valid_loss = np.inf
        self.current_fold = fold

    def __on_fit_end(self) -> None:
        free_gpu_memory(
            self.optimizer,
            self.scheduler,
            self.history_dict["valid_trues"],
            self.history_dict["valid_logits"],
            self.history_dict["valid_preds"],
            self.history_dict["valid_probs"],
        )

    def __train_one_epoch(self, train_loader: DataLoader, epoch: int) -> None:
        """Train one epoch of the model."""
        curr_lr = self.get_lr(self.optimizer)
        train_start_time = time.time()

        # set to train mode
        self.model.train()

        train_bar = tqdm(train_loader)

        # Iterate over train batches
        for _step, batch in enumerate(train_bar, start=1):
            # unpack - note that if BCEWithLogitsLoss, dataset should do view(-1,1) and not here.
            inputs, targets = batch
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            _batch_size = inputs.shape[0]  # unused for now

            with torch.cuda.amp.autocast(
                enabled=self.train_params.use_amp,
                dtype=torch.float16,
                cache_enabled=True,
            ):
                logits = self.model(inputs)  # Forward pass logits
                curr_batch_train_loss = self.computer_criterion(
                    targets,
                    logits,
                    criterion_params=self.pipeline_config.criterion_params,
                    stage="train",
                )
            self.optimizer.zero_grad()  # reset gradients

            if self.scaler is not None:
                self.scaler.scale(curr_batch_train_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                curr_batch_train_loss.backward()  # Backward pass
                self.optimizer.step()  # Update weights using the optimizer

            # Update loss metric, every batch is diff
            self.train_batch_dict["train_loss"] = curr_batch_train_loss.item()
            # train_bar.set_description(f"Train. {metric_monitor}")

            _y_train_prob = get_sigmoid_softmax(self.pipeline_config)(logits)

            _y_train_pred = torch.argmax(_y_train_prob, dim=1)

            self.invoke_callbacks("on_train_batch_end")

        self.invoke_callbacks("on_train_loader_end")
        # total time elapsed for this epoch
        train_time_elapsed = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - train_start_time)
        )
        self.logger.info(
            f"\n[RESULT]: Train. Epoch {epoch}:"
            f"\nAvg Train Summary Loss: {self.train_epoch_dict['train_loss']:.3f}"
            f"\nLearning Rate: {curr_lr:.5f}"
            f"\nTime Elapsed: {train_time_elapsed}\n"
        )
        self.train_history_dict = {**self.train_epoch_dict}
        self.invoke_callbacks("on_train_epoch_end")

    def __valid_one_epoch(self, valid_loader: DataLoader, epoch: int) -> None:
        """Validate the model on the validation set for one epoch.
        Args:
            valid_loader (torch.utils.data.DataLoader): The validation set dataloader.
        Returns:
            Dict[str, np.ndarray]:
                valid_loss (float): The validation loss for each epoch.
                valid_trues (np.ndarray): The ground truth labels for each validation set. shape = (num_samples, 1)
                valid_logits (np.ndarray): The logits for each validation set. shape = (num_samples, num_classes)
                valid_preds (np.ndarray): The predicted labels for each validation set. shape = (num_samples, 1)
                valid_probs (np.ndarray): The predicted probabilities for each validation set. shape = (num_samples, num_classes)
        """
        val_start_time = time.time()  # start time for validation

        self.model.eval()  # set to eval mode

        valid_bar = tqdm(valid_loader)

        valid_logits, valid_trues, valid_preds, valid_probs = [], [], [], []

        with torch.no_grad():
            for _step, batch in enumerate(valid_bar, start=1):
                # unpack
                inputs, targets = batch
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()  # reset gradients

                logits = self.model(inputs)  # Forward pass logits

                # get batch size, may not be same as params.batch_size due to whether drop_last in loader is True or False.
                _batch_size = inputs.shape[0]

                # TODO: Refer to my RANZCR notes on difference between Softmax and Sigmoid with examples.
                y_valid_prob = get_sigmoid_softmax(self.pipeline_config)(logits)
                y_valid_pred = torch.argmax(y_valid_prob, axis=1)

                curr_batch_val_loss = self.computer_criterion(
                    targets,
                    logits,
                    criterion_params=self.pipeline_config.criterion_params,
                    stage="valid",
                )
                # Update loss metric, every batch is diff
                self.valid_batch_dict["valid_loss"] = curr_batch_val_loss.item()

                # valid_bar.set_description(f"Validation. {metric_monitor}")

                self.invoke_callbacks("on_valid_batch_end")
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
        _, valid_metrics_dict = self.get_classification_metrics(
            valid_trues, valid_preds, valid_probs
        )

        self.invoke_callbacks("on_valid_loader_end")

        # total time elapsed for this epoch
        valid_elapsed_time = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - val_start_time)
        )
        self.logger.info(
            f"\n[RESULT]: Validation. Epoch {epoch}:"
            f"\nAvg Val Summary Loss: {self.valid_epoch_dict['valid_loss']:.3f}"
            f"\nAvg Val Accuracy: {valid_metrics_dict['val_Accuracy']:.3f}"
            f"\nAvg Val Macro AUROC: {valid_metrics_dict['val_AUROC']:.3f}"
            f"\nTime Elapsed: {valid_elapsed_time}\n"
        )
        # here self.valid_epoch_dict only has valid_loss, we update the rest
        self.valid_epoch_dict.update(
            {
                "valid_trues": valid_trues,
                "valid_logits": valid_logits,
                "valid_preds": valid_preds,
                "valid_probs": valid_probs,
                "valid_elapsed_time": valid_elapsed_time,
            }
        )  # FIXME: potential difficulty in debugging since valid_epoch_dict is called in metrics meter
        self.valid_epoch_dict.update(valid_metrics_dict)
        # temporary stores current valid epochs info
        # FIXME: so now valid epoch dict and valid history dict are the same lol.
        self.valid_history_dict = {**self.valid_epoch_dict, **valid_metrics_dict}

        # TODO: after valid epoch ends, for example, we need to call
        # our History callback to save the metrics into a list.
        self.invoke_callbacks("on_valid_epoch_end")


    @staticmethod
    def get_optimizer(
        model,
        optimizer_params: Dict[str, Any],
    ) -> torch.optim.Optimizer:
        """Get the optimizer for the model.
        Note:
            Do not invoke self.model directly in this call as it may affect model initalization.
            https://stackoverflow.com/questions/70107044/can-i-define-a-method-as-an-attribute
        """
        return getattr(torch.optim, optimizer_params.optimizer)(
            model.parameters(), **optimizer_params.optimizer_params
        )

    @staticmethod
    def get_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_params: Dict[str, Any],
    ) -> torch.optim.lr_scheduler:
        """Get the scheduler for the optimizer."""
        return getattr(torch.optim.lr_scheduler, scheduler_params.scheduler)(
            optimizer=optimizer, **scheduler_params.scheduler_params
        )
