# Copyright 2017-2018 Fizyr (https://fizyr.com)
#
# Modifications copyright 2021 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Functions to control weights initialisation
"""


# import keras
from typing import Any, Dict
import math
import numpy as np
from tensorflow import keras


class PriorProbability(keras.initializers.Initializer):
    """ Apply a prior probability to the weights.
    """

    def __init__(self, probability: float = 0.01) -> None:
        self.probability = probability

    def get_config(self) -> Dict[str, Any]:
        return {
            'probability': self.probability
        }

    def __call__(self, shape: np.array, dtype: np.dtype = None) -> np.ndarray:
        # set bias to -log((1 - p)/p) for foreground
        result = np.ones(shape, dtype=np.float32) * - \
            math.log((1 - self.probability) / self.probability)

        return result
