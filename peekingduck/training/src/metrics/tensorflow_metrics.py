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

from typing import List
from omegaconf import DictConfig
from src.metrics.base import MetricsAdapter

class TensorflowMetrics(MetricsAdapter):

    def get_metric(self, metric_name: str, parameters: DictConfig = {}):
        return getattr(tf.keras.metrics, metric_name)(**parameters) if len(parameters) > 0 else getattr(tf.keras.metrics, metric_name)()


    def get_metrics(self, metrics: List = []) -> List[tf.keras.metrics.Metric]:
        metrics_collection = []
        for metric in metrics:
            try:
                if type(metric) is DictConfig:
                    for mkey, mval in metric.items():
                        metrics_collection.append( self.get_metric(mkey, mval) )
                elif type(metric) is str:
                    metrics_collection.append( self.get_metric(metric) )
                else:
                    raise TypeError
            except NotImplementedError:
                raise NotImplementedError
        return metrics_collection