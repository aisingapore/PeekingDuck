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

"""Metrics Meter Callback.

Batch/Step 1: curr_batch_train_loss = 10 -> average_cumulative_train_loss = (10-0)/1 = 10
Batch/Step 2: curr_batch_train_loss = 12 -> average_cumulative_train_loss =
    10 + (12-10)/2 = 11 (Basically (10+12)/2=11)
Essentially, average_cumulative_train_loss = loss over all batches / batches
average_cumulative_train_loss += (
    curr_batch_train_loss.detach().item() - average_cumulative_train_loss
) / (step)

Note after 1 full loop epoch,
the model has traversed through all batches in the dataloader.
So, the average score is the average of all batches in the dataloader.
for eg, if train set has 1000 samples and batch size is 100,
then the model will have traversed through 10 batches in 1 epoch.
then the cumulative count is "step" which is 10 in this case.
the cumulative metric score is the sum of all the metric scores of all batches.
so add up all the metric scores of all batches and divide by the cumulative count.
this is the average score of all batches in 1 epoch.
"""
from collections import defaultdict
from typing import Any, List

from src.callbacks.base import Callback
from src.callbacks.order import CallbackOrder
from src.trainer.base import Trainer


class MetricMeter(Callback):
    """Monitor Metrics.
    https://github.com/pytorch/vision/blob/main/references/classification/utils.py
    """

    # this must same as.
    stats_to_track: List[str] = ["train_loss", "valid_loss", "train_acc", "valid_acc"]

    # pylint: disable=attribute-defined-outside-init,dangerous-default-value
    def __init__(
        self, float_precision: int = 3, stats_to_track: List = stats_to_track
    ) -> None:
        """init"""
        super().__init__(order=CallbackOrder.METRICMETER)
        self.float_precision = float_precision
        self.stats_to_track = stats_to_track
        self.reset()

    def reset(self) -> None:
        """To check PyTorch's code for resetting the metric monitor."""
        self.metrics_dict: defaultdict = defaultdict(
            lambda: {
                "cumulative_metric_score": 0.0,
                "cumulative_count": 0.0,
                "average_score": 0.0,
            }
        )

    def on_train_loader_end(self, trainer: Trainer) -> None:
        """Called when the train loader ends."""
        # this loss is the average loss over the entire epoch
        trainer.epoch_dict["train"]["train_loss"] = self.metrics_dict["train_loss"][
            "average_score"
        ]
        # reset metrics dict since we are going to next epoch and will not want to double count
        self.reset()

    def on_train_batch_end(self, trainer: Trainer) -> None:
        """on_train_batch_end"""
        self._update("train_loss", trainer.epoch_dict["train"]["batch_loss"])

    def on_valid_loader_end(self, trainer: Trainer) -> None:
        """this loss is the average loss over the entire epoch"""
        trainer.epoch_dict["validation"]["valid_loss"] = self.metrics_dict[
            "valid_loss"
        ]["average_score"]
        # reset metrics dict since we are going to next epoch and will not want to double count
        self.reset()

    def on_valid_batch_end(self, trainer: Trainer) -> None:
        """on_valid_batch_end"""
        self._update("valid_loss", trainer.epoch_dict["validation"]["batch_loss"])

    def _update(self, metric_name: str, metric_score: Any) -> None:
        """To check PyTorch's code for updating the loss meter."""
        metric = self.metrics_dict[metric_name]

        metric["cumulative_metric_score"] += metric_score
        metric["cumulative_count"] += 1
        metric["average_score"] = (
            metric["cumulative_metric_score"] / metric["cumulative_count"]
        )

    # pylint: disable=consider-using-f-string
    def __str__(self) -> str:
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name,
                    avg=metric["average_score"],
                    float_precision=self.float_precision,
                )
                for (metric_name, metric) in self.metrics_dict.items()
            ]
        )


class AverageLossMeter(MetricMeter):
    """Computes and stores the average and current loss."""

    # pylint: disable=attribute-defined-outside-init
    def __init__(self) -> None:
        """init"""
        super().__init__()
        self.reset()

    def reset(self) -> None:
        """To check PyTorch's code for resetting the loss meter."""

        self.curr_batch_avg_loss = 0.0
        self.avg = 0.0
        self.running_total_loss = 0.0
        self.count = 0.0

    def update(self, curr_batch_avg_loss: float, batch_size: int) -> None:
        """To check PyTorch's code for updating the loss meter.
        Args:
            curr_batch_avg_loss (float): _description_
            batch_size (str): _description_
        """
        self.curr_batch_avg_loss = curr_batch_avg_loss
        self.running_total_loss += curr_batch_avg_loss * batch_size
        self.count += batch_size
        self.avg = self.running_total_loss / self.count
