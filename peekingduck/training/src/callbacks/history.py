"""Implements Model History.
NOTE:
    1. self.history is instantiated on_trainer_start and we do not need to create
        it in trainer anymore.
    2. After everything, we assign self.history to trainer.history so we can call
        trainer.history for results.
"""
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List

from src.callbacks.default_callbacks import Callback
from src.trainer.default_trainer import Trainer


class History(Callback):
    """Class to log metrics to console and save them to a CSV file."""

    def __init__(self) -> None:
        """Constructor class for History Class."""
        super().__init__()
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
        self._update(history=trainer.history_dict)

    def on_valid_epoch_end(self, trainer: Trainer) -> None:
        """Method to update history object at the end of every epoch."""
        self._update(history=trainer.history_dict)

    def on_trainer_end(self, trainer: Trainer) -> None:
        """Method assigns accumulated history to history attribute
        back to Trainer class."""
        trainer.history = self.history

    def _update(self, history: Dict[str, Any]) -> None:
        """Updates the history object with the latest metrics."""
        for key in history:
            if key not in self.history:
                self.history[key] = [history.get(key)]
            else:
                self.history[key].append(history.get(key))
