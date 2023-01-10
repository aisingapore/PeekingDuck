"""Transforms for data augmentation."""
from abc import ABC, abstractmethod
from typing import Any

import torchvision.transforms as T

from omegaconf import DictConfig
from hydra.utils import instantiate

from src.transforms.base import Transforms


class ImageClassificationTransforms(Transforms):
    """General Image Classification Transforms."""

    def __init__(self, pipeline_config: DictConfig) -> None:
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config

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
