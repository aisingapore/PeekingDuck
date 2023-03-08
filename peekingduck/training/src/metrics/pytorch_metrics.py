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

import torchmetrics
from torchmetrics.classification.stat_scores import (
    MulticlassStatScores,
)  # for type hinting, referenced from PyTorch Lightning source code
from torchmetrics import MetricCollection

import torch
from typing import Any, Union
from omegaconf import DictConfig


class PytorchMetrics:
    """pytorch metrics"""

    @classmethod
    def get_metric(
        cls, task: str, num_classes: int, metric: Union[str, DictConfig]
    ) -> MulticlassStatScores:  # the metric can be a dict or list
        """
        Refer to TorchMetrics implementation
        """
        if isinstance(metric, str):
            torch_metric = getattr(torchmetrics, metric)(
                num_classes=num_classes, task=task
            )

        elif isinstance(metric, DictConfig):
            for metric_name, metric_params in metric.items():
                torch_metric = getattr(torchmetrics, str(metric_name))(
                    num_classes=num_classes, task=task, **metric_params
                )

        else:
            raise TypeError(f"Unknown metric type {type(metric)}")

        return torch_metric

    @classmethod
    def get_metrics(
        cls, task: str, num_classes: int, metric_list: list
    ) -> MetricCollection:
        metric_collection_list = [
            cls.get_metric(task, num_classes, metric) for metric in metric_list
        ]
        metrics_collection: MetricCollection = MetricCollection(metric_collection_list)
        return metrics_collection

    @staticmethod
    def get_classification_metrics(
        metrics: MetricCollection,
        y_trues: torch.Tensor,
        y_preds: torch.Tensor,
        y_probs: torch.Tensor,
        prefix: str = "train",
    ) -> Any:
        """
        Calculate metrics
        [summary]
        # https://ghnreigns.github.io/reighns-ml-website/supervised_learning/classification/breast_cancer_wisconsin/Stage%206%20-%20Modelling%20%28Preprocessing%20and%20Spot%20Checking%29/
        Args:
            y_trues (torch.Tensor): dtype=[torch.int64], shape=(num_samples, 1); (May be float if using BCEWithLogitsLoss)
            y_preds (torch.Tensor): dtype=[torch.int64], shape=(num_samples, 1);
            y_probs (torch.Tensor): dtype=[torch.float32], shape=(num_samples, num_classes);
            mode (str, optional): [description]. Defaults to "valid".
        """
        epoch_metrics = metrics.clone(prefix=str(prefix) + "_")
        return epoch_metrics(y_probs, y_trues.flatten())
