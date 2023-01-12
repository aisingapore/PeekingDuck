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
from src.metrics.base import MetricsAdapter

class tensorflowMetrics(MetricsAdapter):
    def __init__(self) -> None:
        pass

    def setup(self,
            task: str = "multiclass",
            num_classes: int = 2,
            metrics: List[str] = None,
            framework: str = "pytorch",
        ) -> List:

        self.num_classes = num_classes
        self.task = task
        self.framework = framework
        self.metrics = metrics
        return []

    def create_collection(self):
        pass
