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

"""Model Checkpoint Callback.

Reference:
    1. pytorch_lightning/callbacks/model_checkpoint.py
    2. torchflare/callbacks/model_checkpoint.py

Logic:
    1. This is called after each validation epoch.
    2. We check if the current epoch score is better than the best score.
    3. If it is, we save the model.
    4. Additional logics such as save every batch, save top k models, etc
        can be enhancements. As currently we are saving all weights sequentially.

Tests:
    1. Test if the model is saved when the score improves. For example,
        if the mode is max, and metric name is val_Accuracy, then if the
        model was trained for 6 epochs, with val_Accuracy of [0.1, 0.2, 0.3, 0.4, 0.3, 0.6],
        then the model should not save epoch 5 since the score of 0.3 is not better than
        0.4.

TODO:
1. Currently, monitored_metric is a param in GlobalTrainParams, and therefore
  the `mode` and `monitor` are already set. So, passing in as
  arguments in `__init__` is redundant. However, we can still keep it as
  it seems clear to the advanced users that this is a callback that can
  take in these arguments.
  Now, the `best_val_score` is initiated from the `mode` argument, if we use our
  `trainer_config`, then we can define it as
  self.best_val_score = -math.inf if trainer.monitored_metric["mode"] == "max" else math.inf

2. A corollary is if we need to put `model_artifacts_dir` in the argument here as well?
  Currently we just `trainer.trainer_config.stores.model_artifacts_dir` to get it.

3. Support for `save_top_k` and `save_every_n_epochs` can be added.

4. Support for saving multiple metrics, i.e. best val_loss and best val_acc.

5. Currently initiating attributes in `__init__` and `on_train_start`, I think it may not
    be the best practice? But otherwise it makes sense to initiate some "when trainer starts".

6. Put all callbacks to trainer_config.callbacks.

FIXME:

1. Not elegant to assign to dict like this in `on_valid_epoch_end`.

Appendix:
Save the weights and states for the best evaluation metric and also the OOF scores.

valid_trues -> oof_trues: np.array of shape [num_samples, 1] and represent the true
    labels for each sample in current fold.
    i.e. oof_trues.flattened()[i] = true label of sample i in current fold.

valid_logits -> oof_logits: np.array of shape [num_samples, num_classes] and
    represent the logits for each sample in current fold.
    i.e. oof_logits[i] = [logit_of_sample_i_in_current_fold_for_class_0,
                            logit_of_sample_i_in_current_fold_for_class_1, ...]

valid_preds -> oof_preds: np.array of shape [num_samples, 1] and represent the
    predicted labels for each sample in current fold.
    i.e. oof_preds.flattened()[i] = predicted label of sample i in current fold.

valid_probs -> oof_probs: np.array of shape [num_samples, num_classes] and represent the
    probabilities for each sample in current fold. i.e. first row is
    the probabilities of the first class.
    i.e. oof_probs[i] = [probability_of_sample_i_in_current_fold_for_class_0,
                            probability_of_sample_i_in_current_fold_for_class_1, ...]
"""
from pathlib import Path
from typing import Any, Dict

import torch

from src.callbacks.base import Callback
from src.callbacks.order import CallbackOrder
from src.trainer.base import Trainer
from src.utils.callback_utils import init_improvement


class ModelCheckpoint(Callback):
    """Callback for Checkpointing your model.
    Referenced from torchflare.

    Args:
        mode: One of {"min", "max"}.
            In min mode, training will stop when the quantity monitored has stopped decreasing
            in "max" mode it will stop when the quantity monitored has stopped increasing.
        monitor: Name of the metric to monitor, should be one of the keys in metrics list.

    Raises:
        ValueError if monitor does not start with prefix ``val_`` or ``train_``.

    Example:
        .. code-block::

            from pkd.callbacks.model_checkpoint import ModelCheckpoint
            model_checkpoint = ModelCheckpoint(mode="max", monitor="val_Accuracy")
    """

    def __init__(
        self,
        mode: str,
        monitor: str,
        min_delta: float = 1e-6,
    ) -> None:
        """Constructor for ModelCheckpoint class."""
        super().__init__(order=CallbackOrder.METRICMETER)

        if monitor.startswith("train_") or monitor.startswith("val_"):
            self.monitor = monitor
        else:
            raise ValueError("Monitor must have a prefix either train_ or val_.")

        self.mode = mode
        self.min_delta = min_delta
        self.improvement = init_improvement(mode=self.mode, min_delta=self.min_delta)
        self.best_val_score = 0.0
        self.state_dict: Dict[str, Any] = {}
        self.model_artifacts_dir = ""

    @staticmethod
    def save_checkpoint(state_dict: Dict[str, Any], model_artifacts_path: Path) -> None:
        """Method to save the state dictionaries of model, optimizer,etc."""
        torch.save(state_dict, model_artifacts_path)

    def on_trainer_start(self, trainer: Trainer) -> None:
        """Initialize the best score as either -inf or inf depending on mode."""
        self.improvement, self.best_val_score = init_improvement(
            mode=self.mode, min_delta=self.min_delta
        )
        self.state_dict = {
            "model_state_dict": None,
            "optimizer_state_dict": None,
            "scheduler_state_dict": None,
            "epoch": None,
            "best_score": None,
            "model_artifacts_path": "",
        }
        self.model_artifacts_dir = (
            trainer.model_artifacts_dir
            if trainer.model_artifacts_dir is not None
            else ""
        )

    def on_valid_epoch_end(self, trainer: Trainer) -> None:
        """Method to save best model depending on the monitored quantity."""
        valid_score = trainer.epoch_dict["validation"]["metrics"].get(self.monitor)

        if self.improvement(
            curr_epoch_score=valid_score, curr_best_score=self.best_val_score
        ):
            self.best_val_score = valid_score

            self.state_dict["model_state_dict"] = trainer.model.state_dict()
            self.state_dict["optimizer_state_dict"] = trainer.optimizer.state_dict()
            self.state_dict["scheduler_state_dict"] = (
                trainer.scheduler.state_dict()
                if not trainer.scheduler is None
                else None
            )
            self.state_dict["epoch"] = trainer.current_epoch
            self.state_dict["best_score"] = self.best_val_score

            if self.model_artifacts_dir != "":
                model_artifacts_path = (
                    Path(self.model_artifacts_dir)
                    .joinpath(
                        f"{trainer.train_params.model_name}_best_{self.monitor}"
                        f"_fold_{trainer.current_fold}_epoch{trainer.current_epoch}.pt",
                    )
                    .as_posix()
                )
                self.state_dict["model_artifacts_path"] = model_artifacts_path
                self.save_checkpoint(self.state_dict, Path(model_artifacts_path))
