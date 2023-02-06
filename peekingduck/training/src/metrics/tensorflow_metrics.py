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

from typing import List
import tensorflow as tf
from src.metrics.base import MetricsAdapter

class tensorflowMetrics(MetricsAdapter):
    def __init__(self) -> None:
        pass

    def setup(self,
            task: str = "multiclass",
            num_classes: int = 2,
            metrics: List[str] = None,
            framework: str = "pytorch",
        ) -> None:

        self.num_classes = num_classes
        self.task = task
        self.framework = framework
        self.metrics = metrics
        self.metrics_collection = []

    def accuracy(self):
        return tf.keras.metrics.Accuracy(name='accuracy', dtype=None)

    def precision(self):
        return  tf.keras.metrics.Precision(thresholds=None, top_k=None, class_id=None, name=None, dtype=None)

    def recall(self):
        return  tf.keras.metrics.Recall(thresholds=None, top_k=None, class_id=None, name=None, dtype=None)

    def f1_score(self):
        pass

    def auroc(self):
        return tf.keras.metrics.AUC(num_thresholds=200, curve='ROC', summation_method='interpolation', name=None, dtype=None, thresholds=None, multi_label=False, num_labels=None, label_weights=None, from_logits=False)

    def create_collection(self) -> List[tf.keras.metrics.Metric]:
        for metric in self.metrics:
            try:
                self.metrics_collection.append(getattr(self, metric)())
            except NotImplementedError:
                pass
        return self.metrics_collection