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

from src.trainer.base import Trainer
from src.callbacks.base import init_callbacks
from src.callbacks.events import EVENTS
from src.model.pytorch_base import PTModel
from src.metrics.pytorch_metrics import PytorchMetrics
from src.utils.general_utils import free_gpu_memory  # , init_logger

import logging
from configs import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name


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
        self.logger = logger
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
        self.train_params = self.trainer_config.global_train_params

        self.model_artifacts_dir = self.trainer_config.stores.model_artifacts_dir
        self.device = device
        self.callbacks = init_callbacks(callbacks_config[self.framework])
        metrics_adapter = PytorchMetrics(
            task=data_config.dataset.classification_type,
            num_classes=data_config.dataset.num_classes,
            metrics=metrics_config[self.framework].evaluate,
        )
        self.metrics = metrics_adapter.create_collection()

        self.model: PTModel = instantiate(
            config=model_config[self.framework].model_type,
            cfg=model_config[self.framework],
            _recursive_=False,
        ).to(self.device)

        self.optimizer = self.get_optimizer(
            model=self.model,
            optimizer_params=self.trainer_config.optimizer_params,
        )
        self.scheduler = self.get_scheduler(
            optimizer=self.optimizer,
            scheduler_params=self.trainer_config.scheduler_params,
        )

        if self.train_params.use_amp:  # TODO
            # https://pytorch.org/docs/stable/notes/amp_examples.html
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

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
        self._invoke_callbacks(EVENTS.ON_TRAINER_START.value)

    def _set_dataloaders(
        self,
        train_dl: DataLoader,
        validation_dl: DataLoader,
    ) -> None:
        """Initialise Dataloader Variables"""
        self.train_loader = train_dl
        self.validation_loader = validation_dl

    def _train_setup(self, inputs, fold: int) -> None:
        self.logger.info(f"Fold {fold} started")
        # show model summary
        self.model.model_summary(inputs.shape)
        self.best_valid_loss = np.inf
        self.current_fold = fold

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

            self._run_train_epoch(self.train_loader, epoch)
            self._run_validation_epoch(self.validation_loader, epoch)

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

            self.epoch_dict["train"]["epoch"] = epoch
            self.epoch_dict["validation"]["epoch"] = epoch
            self.current_epoch += 1

    def _run_train_epoch(self, train_loader: DataLoader, epoch: int) -> None:
        """Train one epoch of the model."""
        train_start_time = time.time()
        self._invoke_callbacks(EVENTS.ON_TRAIN_EPOCH_START.value)

        curr_lr = self.get_lr(self.optimizer)
        # set to train mode
        self.model.train()

        train_bar = tqdm(train_loader)

        self._invoke_callbacks(EVENTS.ON_TRAIN_LOADER_START.value)
        # Iterate over train batches
        for _, batch in enumerate(train_bar, start=1):
            self._invoke_callbacks(EVENTS.ON_TRAIN_BATCH_START.value)

            # unpack - note that if BCEWithLogitsLoss, dataset should do view(-1,1) and not here.
            inputs, targets = batch
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # with torch.cuda.amp.autocast(  # TODO
            #     enabled=self.train_params.use_amp,
            #     dtype=torch.float16,
            #     cache_enabled=True,
            # ):
            logits = self.model(inputs)  # Forward pass logits
            curr_batch_train_loss = self.compute_criterion(
                targets,
                logits,
                criterion_params=self.trainer_config.criterion_params,
                stage="train",
            )

            self.optimizer.zero_grad()  # reset gradients

            # if self.scaler is not None:
            #     self.scaler.scale(curr_batch_train_loss).backward()
            #     self.scaler.step(self.optimizer)
            #     self.scaler.update()
            # else:
            curr_batch_train_loss.backward()  # Backward pass
            self.optimizer.step()  # Update weights using the optimizer

            # Update loss metric, every batch is diff
            self.batch_dict["train_loss"] = curr_batch_train_loss.item()
            _y_train_prob = get_sigmoid_softmax(self.trainer_config)(logits)
            _y_train_pred = torch.argmax(_y_train_prob, dim=1)

            self._invoke_callbacks(EVENTS.ON_TRAIN_BATCH_END.value)

        self._invoke_callbacks(EVENTS.ON_TRAIN_LOADER_END.value)

        self.history_dict["train"] = {**self.epoch_dict["train"]}
        # total time elapsed for this epoch
        train_time_elapsed = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - train_start_time)
        )
        self.logger.info(
            f"\n[RESULT]: Train. Epoch {epoch}:"
            f"\nAvg Train Summary Loss: {self.epoch_dict['train']['train_loss']:.3f}"
            f"\nLearning Rate: {curr_lr:.5f}"
            f"\nTime Elapsed: {train_time_elapsed}\n"
        )
        self._invoke_callbacks(EVENTS.ON_TRAIN_EPOCH_END.value)

    def _run_validation_epoch(self, validation_loader: DataLoader, epoch: int) -> None:
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

                curr_batch_val_loss = self.compute_criterion(
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
        _, valid_metrics_dict = self._get_classification_metrics(
            valid_trues, valid_preds, valid_probs
        )

        self._invoke_callbacks(EVENTS.ON_VALID_LOADER_END.value)

        # total time elapsed for this epoch
        valid_elapsed_time = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - val_start_time)
        )
        self.epoch_dict['validation'].update(
            {
                "valid_trues": valid_trues,
                "valid_logits": valid_logits,
                "valid_preds": valid_preds,
                "valid_probs": valid_probs,
                "valid_elapsed_time": valid_elapsed_time,
            }
        )
        # FIXME: potential difficulty in debugging since epoch_dict is called in metrics meter
        self.epoch_dict["validation"].update(valid_metrics_dict)
        self.history_dict["validation"] = {**self.epoch_dict["validation"], **valid_metrics_dict}
        self.logger.info(
            f"\n[RESULT]: Validation. Epoch {epoch}:"
            f"\nAvg Val Summary Loss: {self.epoch_dict['validation']['valid_loss']:.3f}"
            f"\nAvg Val Accuracy: {valid_metrics_dict['val_MulticlassAccuracy']:.3f}"
            f"\nAvg Val Macro AUROC: {valid_metrics_dict['val_MulticlassAUROC']:.3f}"
            f"\nTime Elapsed: {valid_elapsed_time}\n"
        )
        self._invoke_callbacks(EVENTS.ON_VALID_EPOCH_END.value)

    def _invoke_callbacks(self, event_name: str) -> None:
        """Invoke the callbacks."""
        for callback in self.callbacks:
            try:
                getattr(callback, event_name)(self)
            except NotImplementedError:
                pass

    def _get_classification_metrics(
        self,
        y_trues: torch.Tensor,
        y_preds: torch.Tensor,
        y_probs: torch.Tensor,
        # mode: str = "valid",
    ):
        """[summary]
        # https://ghnreigns.github.io/reighns-ml-website/supervised_learning/classification/breast_cancer_wisconsin/Stage%206%20-%20Modelling%20%28Preprocessing%20and%20Spot%20Checking%29/
        Args:
            y_trues (torch.Tensor): dtype=[torch.int64], shape=(num_samples, 1); (May be float if using BCEWithLogitsLoss)
            y_preds (torch.Tensor): dtype=[torch.int64], shape=(num_samples, 1);
            y_probs (torch.Tensor): dtype=[torch.float32], shape=(num_samples, num_classes);
            mode (str, optional): [description]. Defaults to "valid".
        """

        train_metrics = self.metrics.clone(prefix="train_")
        valid_metrics = self.metrics.clone(prefix="val_")

        # FIXME: currently train and valid give same results, since this func call takes in
        # y_trues, etc from valid_one_epoch.
        train_metrics_results = train_metrics(y_probs, y_trues.flatten())
        train_metrics_results_df = pd.DataFrame.from_dict([train_metrics_results])

        valid_metrics_results = valid_metrics(y_probs, y_trues.flatten())
        valid_metrics_results_df = pd.DataFrame.from_dict([valid_metrics_results])

        # TODO: relinquish this logging duty to a callback or for now in train_one_epoch and valid_one_epoch.
        self.logger.info(
            f"\ntrain_metrics:\n{tabulate(train_metrics_results_df, headers='keys', tablefmt='psql')}\n"
        )
        self.logger.info(
            f'\nvalid_metrics:\n{tabulate(valid_metrics_results_df, headers="keys", tablefmt="psql")}\n'
        )
        return train_metrics_results, valid_metrics_results

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

    @staticmethod
    def compute_criterion(
        y_trues: torch.Tensor,
        y_logits: torch.Tensor,
        criterion_params: Dict[str, Any],
        stage: str,
    ) -> torch.Tensor:
        """Train Loss Function.
        Note that we can evaluate train and validation fold with different loss functions.
        The below example applies for CrossEntropyLoss.
        Args:
            y_trues ([type]): Input - N,C) where N = number of samples and C = number of classes.
            y_logits ([type]): If containing class indices, shape (N) where each value is
                $0 \leq \text{targets}[i] \leq C-10≤targets[i]≤C-1$.
                If containing class probabilities, same shape as the input.
            stage (str): train or valid, sometimes people use different loss functions for
                train and valid.
        """

        if stage == "train":
            loss_fn = getattr(torch.nn, criterion_params.train_criterion)(
                **criterion_params.train_criterion_params
            )
        elif stage == "valid":
            loss_fn = getattr(torch.nn, criterion_params.valid_criterion)(
                **criterion_params.valid_criterion_params
            )
        loss = loss_fn(y_logits, y_trues)
        return loss

    @staticmethod
    def get_lr(optimizer: torch.optim) -> float:
        """Get the learning rate of optimizer for the current epoch.
        Note learning rate can be different for different layers, hence the for loop.
        """
        for param_group in optimizer.param_groups:
            return param_group["lr"]


    def train(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        fold: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Fit the model and returns the history object."""
        self._invoke_callbacks(EVENTS.ON_TRAINER_START.value)

        self._set_dataloaders(train_dl=train_loader, validation_dl=validation_loader)
        inputs, _ = next(iter(train_loader))
        self._train_setup(inputs, fold=fold)  # startup
        self._run_epochs()
        self._train_teardown()  # shutdown

        self._invoke_callbacks(EVENTS.ON_TRAINER_END.value)
        return self.history