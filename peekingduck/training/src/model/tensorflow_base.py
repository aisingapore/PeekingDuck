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

from typing import Tuple, Type
from abc import ABC, abstractmethod

import tensorflow as tf

class Model(ABC):
    """Model Base Class for TensorFlow."""
    
    def __init__(self) -> None:
        self.model_config = None
        self.train_dataset = None
        self.valid_dataset = None
        self.model_name: str
        self.input_shape: Tuple[int, int, int]
        self.learning_rate: float
        self.optimizer: Type[tf.keras.optimizers.Optimizer]
        self.loss: Type[tf.keras.losses.Loss]
        self.metrics: list

    @abstractmethod
    def create_base(self):
        """Create pre-trained base model."""
        raise NotImplementedError
        
    @abstractmethod
    def create_head(self):
        """Create head of the model."""
        raise NotImplementedError
        
    @abstractmethod
    def build_model(self):
        """Build the model with base and head"""
        
    @abstractmethod
    def compile_model(self):
        """Compile model with optimizer, loss and metrics."""
        raise NotImplementedError
    
    @abstractmethod
    def create_model(self):
        """Create model with the build and compile methods."""
        raise NotImplementedError