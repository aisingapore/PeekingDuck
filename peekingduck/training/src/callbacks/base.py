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
# Code of this file is mostly forked from
# [@pytorch_lightning](https://github.com/Lightning-AI/lightning/blob/master/src/
# pytorch_lightning/callbacks/callback.py))

"""Callback base class. This does not require inheritance from ABC as different
child class can have different methods."""

from __future__ import annotations

from typing import TYPE_CHECKING  # This solves circular import type hinting.

if TYPE_CHECKING:
    from src.trainer.base import Trainer


# pylint: disable=too-many-public-methods
class Callback:
    """Callback base class."""

    def __init__(self) -> None:
        """Constructor for Callback base class."""

    def setup(self, trainer: Trainer, stage: str) -> None:
        """Called when fit, validate, test, predict, or tune begins."""

    def teardown(self, trainer: Trainer, stage: str) -> None:
        """Called when fit, validate, test, predict, or tune ends."""

    def on_trainer_start(self, trainer: Trainer) -> None:
        """Called when the trainer: Trainer begins."""

    def on_trainer_end(self, trainer: Trainer) -> None:
        """Called when the trainer: Trainer ends."""

    def on_fit_start(self, trainer: Trainer) -> None:
        """Called AFTER fit begins."""

    def on_fit_end(self, trainer: Trainer) -> None:
        """Called AFTER fit ends."""

    def on_train_batch_start(self, trainer: Trainer) -> None:
        """Called when the train batch begins."""

    def on_train_batch_end(self, trainer: Trainer) -> None:
        """Called when the train batch ends."""

    def on_train_loader_start(self, trainer: Trainer) -> None:
        """Called when the train loader begins."""

    def on_valid_loader_start(self, trainer: Trainer) -> None:
        """Called when the validation loader begins."""

    def on_train_loader_end(self, trainer: Trainer) -> None:
        """Called when the train loader ends."""

    def on_valid_loader_end(self, trainer: Trainer) -> None:
        """Called when the validation loader ends."""

    def on_train_epoch_start(self, trainer: Trainer) -> None:
        """Called when the train epoch begins."""

    def on_train_epoch_end(self, trainer: Trainer) -> None:
        """Called when the train epoch ends."""

    def on_valid_epoch_start(self, trainer: Trainer) -> None:
        """Called when the val epoch begins."""

    def on_valid_epoch_end(self, trainer: Trainer) -> None:
        """Called when the val epoch ends."""

    def on_valid_batch_start(self, trainer: Trainer) -> None:
        """Called when the validation batch begins."""

    def on_valid_batch_end(self, trainer: Trainer) -> None:
        """Called when the validation batch ends."""

    def on_inference_start(self, trainer: Trainer) -> None:
        """Called when the inference begins."""

    def on_inference_end(self, trainer: Trainer) -> None:
        """Called when the inference ends."""
