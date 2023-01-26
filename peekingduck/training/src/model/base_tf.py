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

from abc import ABC, abstractmethod

class Model(ABC):
    """Model Base Class for TensorFlow."""
    
    def __init__(self) -> None:
        self.model_config = None
    
    @abstractmethod
    def create_base(self):
        """Create pre-trained base model."""
        raise NotImplementedError
        
    @abstractmethod
    def create_head(self):
        """Create head of the model."""
        raise NotImplementedError
        
    @abstractmethod
    def create_model(self):
        """Create model with the base and the head."""
        raise NotImplementedError
        
    @abstractmethod
    def compile_model(self):
        """Compile model with optimizer, loss and metrics."""
        raise NotImplementedError