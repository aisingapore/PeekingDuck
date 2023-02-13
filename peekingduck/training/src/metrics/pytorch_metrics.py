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


import inspect
import torchmetrics
from torchmetrics.classification.stat_scores import (
    MulticlassStatScores,
)  # for type hinting, referenced from PyTorch Lightning source code
from torchmetrics import AUROC, Accuracy, MetricCollection, Precision, Recall
from torchmetrics.classification import MulticlassCalibrationError
from typing import List, Dict
from omegaconf import DictConfig
from src.metrics.base import MetricsAdapter


class PytorchMetrics(MetricsAdapter):
    @classmethod
    def get_metrics(cls, task, num_classes, metric_list: list) -> MetricCollection:
        metric_collection_list = []

        for metric in metric_list:
            metric_collection_list.append(cls.get_metric(task, num_classes, metric))
        metrics_collection: MetricCollection = MetricCollection(metric_collection_list)
        return metrics_collection

    @classmethod
    def get_metric(
        cls, task: str, num_classes: int, metric
    ) -> MulticlassStatScores:  # the metric can be a dict or list
        """
        Refer to TorchMetrics implementation
        """
        if type(metric) is str:
            torch_metric = getattr(torchmetrics, metric)(
                num_classes=num_classes, task=task
            )

        elif type(metric) is DictConfig:
            for metric_name, metric_params in metric.items():
                torch_metric = getattr(torchmetrics, metric_name)(
                    num_classes=num_classes, task=task, **metric_params
                )

        else:
            raise TypeError

        return torch_metric
