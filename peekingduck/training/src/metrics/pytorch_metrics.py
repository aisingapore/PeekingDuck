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
from typing import List
from src.metrics.base import MetricsAdapter

class torchMetrics(MetricsAdapter):
    def __init__(self) -> None:
        pass

    def setup(self,
            task: str = "multiclass",
            num_classes: int = 2,
            metrics: List[str] = None,
            framework: str = "pytorch",
        ) -> MetricCollection:

        self.num_classes = num_classes
        self.task = task
        self.framework = framework
        self.metrics = metrics
        return self.create_collection()

    def accuracy(self):
        return Accuracy(
            task=self.task, num_classes=self.num_classes
        )

    def precision(self):
        return  Precision(
            task=self.task, num_classes=self.num_classes, average="macro"
        )

    def recall(self):
        return  Recall(
            task=self.task, num_classes=self.num_classes, average="macro"
        )

    def f1_score(self):
        pass

    def auroc(self):
        return AUROC(
            task=self.task, num_classes=self.num_classes, average="macro"
        )

    def multiclass_calibration_error(self):
        return  MulticlassCalibrationError(
            num_classes=self.num_classes
        )

    def create_collection(self):

        metricList = {}
        for metric in self.metrics:
            try:
                metricList[metric] = getattr(self, metric)()
            except NotImplementedError:
                pass

        self.metrics_collection = MetricCollection(list(metricList.values()))
        return self.metrics_collection