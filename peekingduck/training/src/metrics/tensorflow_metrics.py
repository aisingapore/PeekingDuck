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

import tensorflow as tf

from typing import List, Dict
from omegaconf import DictConfig
from src.metrics.base import MetricsAdapter

class TensorflowMetrics(MetricsAdapter):
    def __init__(self,
            metrics: List[str] = None,
    ) -> None:
        self.metrics_collection = []
        self.metrics = metrics
        for metric in self.metrics:
            try:
                if type(metric) is DictConfig:
                    for mkey, mval in metric.items():
                        self.metrics_collection.append( getattr(self, mkey)(mval) )
                elif type(metric) is str:
                    self.metrics_collection.append( getattr(self, metric)() )
                else:
                    raise TypeError
            except NotImplementedError:
                raise NotImplementedError

    def accuracy(self, parameters: Dict = {}):
        return tf.keras.metrics.Accuracy(**parameters)

    def categorical_accuracy(self, parameters: Dict = {}):
        return tf.keras.metrics.CategoricalAccuracy(**parameters)

    def sparse_categorical_accuracy(self, parameters: Dict = {}):
        return tf.keras.metrics.SparseCategoricalAccuracy(**parameters)

    def precision(self, parameters: Dict = {}):
        return  tf.keras.metrics.Precision(**parameters)

    def recall(self, parameters: Dict = {}):
        return  tf.keras.metrics.Recall(**parameters)

    def auroc(self, parameters: Dict = {}):
        return tf.keras.metrics.AUC(**parameters)

    def get_metrics(self) -> List[tf.keras.metrics.Metric]:
        return self.metrics_collection