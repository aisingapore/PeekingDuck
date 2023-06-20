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

"""Early Stopping Callbacks.

Logic:
    0. We keep track of the following variables:
        - best_score: the best score so far, can be loss or metric.
        - patience: the number of epochs to wait before stopping, if the score
            does not improve.
    1. This is called after each validation epoch.
    2. We check if the current epoch score is better than the best score.
    3. If it is, we change the best score to the current score and reset the patience.
    4. If it is not, we decrement the patience.
    5. If the patience is 0, we stop the training.

"""
import logging

from src.callbacks.base import Callback
from src.callbacks.order import CallbackOrder

from src.trainer.base import Trainer
from src.utils.callback_utils import init_improvement
from configs import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name


class EarlyStopping(Callback):
    """Class for Early Stopping.

    Args:
        mode (str): One of {"min", "max"}. In min mode, training will stop
            when the quantity monitored has stopped decreasing. In "max"
            mode it will stop when the quantity monitored has stopped increasing.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self, mode: str, monitor: str, patience: int = 3, min_delta: float = 1e-6
    ) -> None:
        super().__init__(order=CallbackOrder.EARLYSTOPPING)
        self.mode = mode
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.improvement = init_improvement(mode=self.mode, min_delta=self.min_delta)
        self.best_val_score = None
        self.patience_counter = 0
        self.stop = False

    def on_trainer_start(self, trainer: Trainer) -> None:
        """on_trainer_start"""
        self.improvement, self.best_val_score = init_improvement(
            mode=self.mode, min_delta=self.min_delta
        )
        assert self.improvement is not None, "on_trainer_start: Empty callable."
        self.patience_counter = 0
        self.stop = False
        trainer.stop_training = self.stop  # assign to trainer

    def on_valid_epoch_end(self, trainer: Trainer) -> None:
        """on_valid_epoch_end"""
        assert self.improvement is not None, "on_valid_epoch_end: Empty callable."
        valid_score = trainer.epoch_dict["validation"]["metrics"].get(self.monitor)
        if self.improvement(
            curr_epoch_score=valid_score, curr_best_score=self.best_val_score
        ):
            # update self.best_val_score
            self.best_val_score = valid_score

            # reset patience counter
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            # pylint: disable=logging-fstring-interpolation
            logger.info(
                f"Early Stopping Counter {self.patience_counter} out of {self.patience}"
            )

            if self.patience_counter >= self.patience:
                self.stop = True
                trainer.stop_training = self.stop
                logger.info("Early Stopping!")

    def should_stop(self) -> None:
        """The actual algorithm of early stopping.
        Consider shifting the stop logic here.
        """
