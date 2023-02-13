

import logging
from tabulate import tabulate
from src.trainer.base import Trainer
from src.callbacks.base import Callback
from src.callbacks.order import CallbackOrder

from configs import LOGGER_NAME
logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name

class Logger(Callback):


    def __init__(
        self
    ) -> None:
        """Constructor for Logger class."""
        super().__init__(order=CallbackOrder.LOGGER)
        self.logger = logger

    def on_trainer_start(self, trainer: Trainer) -> None:
        self.logger.info(f"Fold {trainer.current_fold} started")

    def on_train_loader_end(self, trainer: Trainer) -> None:
        self.logger.info(
            f"\ntrain_metrics:\n{tabulate(trainer.metrics_dict['train']['train_metrics_df'], headers='keys', tablefmt='psql')}\n"
        )

    def on_train_epoch_end(self, trainer: Trainer) -> None:
        self.logger.info(
            f"\n[RESULT]: Train. Epoch {trainer.curr_epoch}:"
            f"\nAvg Train Summary Loss: {trainer.epoch_dict['train']['train_loss']:.3f}"
            f"\nLearning Rate: {trainer.curr_lr:.5f}"
            f"\nTime Elapsed: {trainer.train_time_elapsed}\n"
        )

    def on_valid_loader_end(self, trainer: Trainer) -> None:
        self.logger.info(
            f'\nvalid_metrics:\n{tabulate(trainer.metrics_dict["validation"]["valid_metrics_df"], headers="keys", tablefmt="psql")}\n'
        )

    def on_valid_epoch_end(self, trainer: Trainer) -> None:
        self.logger.info(
            f"\n[RESULT]: Validation. Epoch {trainer.curr_epoch}:"
            f"\nAvg Val Summary Loss: {trainer.epoch_dict['validation']['valid_loss']:.3f}"
            f"\nAvg Val Accuracy: {trainer.metrics_dict['validation']['valid_metrics_dict']['val_MulticlassAccuracy']:.3f}"
            f"\nAvg Val Macro AUROC: {trainer.metrics_dict['validation']['valid_metrics_dict']['val_MulticlassAUROC']:.3f}"
            f"\nTime Elapsed: {trainer.valid_elapsed_time}\n"
        )