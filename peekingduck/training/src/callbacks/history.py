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


"""Implements Model History.
NOTE:
    1. self.history is instantiated on_trainer_start and we do not need to create
        it in trainer anymore.
    2. After everything, we assign self.history to trainer.history so we can call
        trainer.history for results.
"""
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List

import wandb

from src.callbacks.base import Callback
from src.callbacks.order import CallbackOrder
from src.trainer.base import Trainer


class History(Callback):
    """Class to log metrics to console and save them to a CSV file."""

    def __init__(self) -> None:
        """Constructor class for History Class."""
        super().__init__(order=CallbackOrder.HISTORY)
        self.history: DefaultDict[str, List[Any]]

    def on_trainer_start(
        self, trainer: Trainer  # pylint: disable=unused-argument
    ) -> None:
        """When the trainer starts, we should initialize the history.
        This is init method of Trainer.
        """
        self.history = defaultdict(list)

    def on_train_epoch_end(self, trainer: Trainer) -> None:
        """Method to update history object at the end of every epoch."""
        self._update(history=trainer.epoch_dict["train"])

    def on_valid_epoch_end(self, trainer: Trainer) -> None:
        """Method to update history object at the end of every epoch."""
        self._update(history=trainer.epoch_dict["validation"])

    def on_trainer_end(self, trainer: Trainer) -> None:
        """Method assigns accumulated history to history attribute
        back to Trainer class."""
        trainer.history = self.history

    def _update(self, history: Dict[str, Any]) -> None:
        """Updates the history object with the latest metrics."""
        wandb.log(history)
        for key in history:
            if key not in self.history:
                self.history[key] = [history.get(key)]
            else:
                self.history[key].append(history.get(key))
