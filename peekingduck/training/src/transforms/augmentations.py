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

import operator
from typing import Any, Dict, Literal

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from omegaconf import DictConfig
from hydra.utils import instantiate

from src.transforms.base import Transforms
from src.config import TF_AVAILABLE

if TF_AVAILABLE:
    import tensorflow as tf
else:
    raise ImportError(
        "Called a tensorflow-specific function but tensorflow is not installed."
    )


class ImageClassificationTransforms(Transforms):
    """General Image Classification Transforms."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.cfg: DictConfig = cfg

    @property
    def train_transforms(self) -> A.Compose:
        """train_transforms"""
        return A.Compose(instantiate(self.cfg.train))

    @property
    def valid_transforms(self) -> A.Compose:
        """valid_transforms"""
        return A.Compose(instantiate(self.cfg.test))

    @property
    def test_transforms(self) -> A.Compose:
        """test_transforms"""
        return A.Compose(instantiate(self.cfg.test))

    @property
    def debug_transforms(self) -> A.Compose:
        """debug_transforms"""
        return self.cfg.transforms.debug_transforms


class TFPreprocessImage(ImageOnlyTransform):
    """Preprocessed numpy.array or a tf.Tensor with type float32.
        The images are converted from RGB to BGR,
        then each color channel is zero-centered with respect to the ImageNet dataset,
        without scaling.
        Refer to
        https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/preprocess_input
    Args:
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self, preprocessor: str = "", always_apply: bool = True, p: float = 1
    ) -> None:
        super().__init__(always_apply, p)
        self.preprocessor: str = preprocessor

    # pylint: disable=unused-argument
    def apply(self, img: list, **kwargs: Dict[str, Any]) -> Any:
        """apply"""
        return operator.attrgetter(self.preprocessor)(tf)(img)

    def get_transform_init_args_names(self) -> Literal["preprocessor"]:
        """get_transform_init_args_names"""
        return "preprocessor"

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """get_params_dependent_on_targets"""
        return params
