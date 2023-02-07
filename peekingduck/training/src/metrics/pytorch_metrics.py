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

from torchmetrics import AUROC, Accuracy, MetricCollection, Precision, Recall
from torchmetrics.classification import MulticlassCalibrationError
from typing import List, Dict
from omegaconf import DictConfig
from src.metrics.base import MetricsAdapter


class PytorchMetrics(MetricsAdapter):
    def __init__(
        self,
        task: str = "multiclass",
        num_classes: int = 2,
        metrics: List[str] = None,
    ) -> None:
        self.metrics_collection = None
        self.num_classes = num_classes
        self.task = task
        self.metrics = metrics
        self.metricList = {}
        for metric in self.metrics:
            try:
                if type(metric) is DictConfig:
                    for mkey, mval in metric.items():
                        self.metricList[metric] = getattr(self, mkey)(mval)
                elif type(metric) is str:
                    self.metricList[metric] = getattr(self, metric)()
                else:
                    raise TypeError
            except NotImplementedError:
                raise NotImplementedError

        for metric in self.metrics:
            try:
                self.metricList[metric] = getattr(self, metric)()
            except NotImplementedError:
                raise NotImplementedError

    def accuracy(self, parameters: Dict = {}):
        return Accuracy(task=self.task, num_classes=self.num_classes, **parameters)

    def precision(self, parameters: Dict = {}):
        return Precision(
            task=self.task, num_classes=self.num_classes, average="macro", **parameters
        )

    def recall(self, parameters: Dict = {}):
        return Recall(
            task=self.task, num_classes=self.num_classes, average="macro", **parameters
        )

    def auroc(self, parameters: Dict = {}):
        return AUROC(
            task=self.task, num_classes=self.num_classes, average="macro", **parameters
        )

    def multiclass_calibration_error(self, parameters: Dict = {}):
        return MulticlassCalibrationError(num_classes=self.num_classes, **parameters)

    def get_metrics(self) -> MetricCollection:
        self.metrics_collection = MetricCollection(list(self.metricList.values()))
        return self.metrics_collection
