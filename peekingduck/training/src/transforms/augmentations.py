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


"""Transforms for data augmentation."""
import torchvision.transforms as T

from omegaconf import DictConfig
from hydra.utils import instantiate

from src.transforms.base import Transforms


class ImageClassificationTransforms(Transforms):
    """General Image Classification Transforms."""

    def __init__(self, pipeline_config: DictConfig) -> None:
        super().__init__(pipeline_config)
        self.pipeline_config: DictConfig = pipeline_config

    @property
    def train_transforms(self) -> T.Compose:
        return T.Compose(instantiate(self.pipeline_config.transform.train))

    @property
    def valid_transforms(self) -> T.Compose:
        return T.Compose(instantiate(self.pipeline_config.transform.train))

    @property
    def test_transforms(self) -> T.Compose:
        return T.Compose(instantiate(self.pipeline_config.transform.train))

    @property
    def debug_transforms(self) -> T.Compose:
        return self.pipeline_config.transforms.debug_transforms
