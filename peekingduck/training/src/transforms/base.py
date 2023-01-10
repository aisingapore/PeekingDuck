"""Transforms for data augmentation."""
from abc import ABC, abstractmethod
from typing import Any

from omegaconf import DictConfig


class Transforms(ABC):
    """Create a Transforms class that can take in albumentations
    and torchvision transforms.
    """

    def __init__(self, pipeline_config: DictConfig) -> None:
        self.pipeline_config = pipeline_config

    @property
    @abstractmethod
    def train_transforms(self):
        """Get the training transforms."""

    @property
    @abstractmethod
    def valid_transforms(self):
        """Get the validation transforms."""

    @property
    def test_transforms(self):
        """Get the test transforms."""

    @property
    def gradcam_transforms(self):
        """Get the gradcam transforms."""

    @property
    def debug_transforms(self):
        """Get the debug transforms."""

    @property
    def test_time_augmentations(self):
        """Get the test time augmentations."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the transforms."""
