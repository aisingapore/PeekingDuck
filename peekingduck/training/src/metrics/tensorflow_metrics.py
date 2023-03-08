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

from typing import Any, Dict, List, Optional, Union
from omegaconf import DictConfig


class TensorflowMetrics:
    """tensorflow metrics"""

    def get_metric(
        self, metric_name: str, parameters: Optional[Union[DictConfig, Dict]] = None
    ) -> tf.keras.metrics.Metric:
        return (
            getattr(tf.keras.metrics, metric_name)(**parameters)
            if parameters is not None and len(parameters) > 0
            else getattr(tf.keras.metrics, metric_name)()
        )

    def get_metrics(self, metrics: List) -> List[Any]:
        return (
            [self.get_metric(**self._validate_config(metric)) for metric in metrics]
            if len(metrics) > 0
            else []
        )

    def _validate_config(self, metric: tf.keras.metrics.Metric) -> Dict[str, Any]:
        if isinstance(metric, DictConfig):
            for mkey, mval in metric.items():
                return {"metric_name": mkey, "parameters": mval}
        elif isinstance(metric, str):
            return {"metric_name": metric}
        else:
            raise TypeError

        return {}
