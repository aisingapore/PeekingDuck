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

from typing import Any, DefaultDict, List

from src.callbacks.base import Callback
from src.trainer.base import Trainer

# pylint: disable=unused-argument
class History(Callback):
    """Callback to record history of training."""

    def __init__(self) -> None:
        """Constructor method."""
        super().__init__()
        self.history: DefaultDict(str, List[Any])

    def on_trainer_start(self, trainer: Trainer) -> None:
        """Initializes history."""
        ...

    def on_train_epoch_end(self, trainer: Trainer) -> None:
        """Updates train history after each epoch."""
        ...

    def on_valid_epoch_end(self, trainer: Trainer) -> None:
        """Updates valid history after each epoch."""
        ...

    def on_trainer_end(self, trainer: Trainer) -> None:
        """Assigns history to trainer."""
        ...

    def _update(self, history: DefaultDict[str, List[Any]]) -> None:
        """Updates history object."""
        ...
