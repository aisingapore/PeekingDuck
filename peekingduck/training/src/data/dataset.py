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


from typing import Any, Optional, Tuple, Union
import numpy as np
from omegaconf import DictConfig

import albumentations as A
import cv2
import pandas as pd

from .. import config

if config.TORCH_AVAILABLE:
    import torch
    from torch import Tensor
    from torch.utils.data import Dataset
    import torchvision.transforms as T
else:
    raise ImportError("Called a torch-specific function but torch is not installed.")

if config.TF_AVAILABLE:
    import tensorflow as tf
else:
    raise ImportError(
        "Called a tensorflow-specific function but tensorflow is not installed."
    )

TransformTypes = Optional[A.Compose]


class PTImageClassificationDataset(Dataset):
    """Template for Image Classification Dataset."""

    def __init__(
        self,
        cfg: DictConfig,
        df: pd.DataFrame = None,
        stage: str = "train",
        transforms: TransformTypes = None,
        **kwargs,
    ) -> None:
        """"""

        super().__init__(**kwargs)
        self.cfg: DictConfig = cfg
        self.df: pd.DataFrame | None = df
        self.stage: str = stage
        self.transforms: A.Compose | None = transforms

        self.image_path = df[cfg.dataset.image_path_col_name].values
        self.targets = df[cfg.dataset.target_col_id].values if stage != "test" else None

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.df)

    def __getitem__(
        self, index: int
    ) -> Union[torch.FloatTensor, Union[torch.FloatTensor, torch.LongTensor]]:
        """Generate one batch of data"""
        image_path: str = self.image_path[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image: Tensor = self.apply_image_transforms(image)

        # Get target for all modes except for test dataset.
        # If test, replace target with dummy ones as placeholder.
        target = self.targets[index] if self.stage != "test" else torch.ones(1)
        # target = self.apply_target_transforms(target)

        # TODO: consider stage to be private since it is only used internally.
        if self.stage in ["train", "valid", "debug"]:
            return image, target
        elif self.stage == "test":
            return image
        else:
            raise ValueError(f"Invalid stage {self.stage}.")

    def apply_image_transforms(self, image: torch.Tensor) -> Tensor:
        """Apply transforms to the image."""
        if self.transforms and isinstance(self.transforms, A.Compose):
            image: Tensor = self.transforms(image=image)["image"]
        elif self.transforms and isinstance(self.transforms, T.Compose):
            image = self.transforms(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)  # convert HWC to CHW
        return image

    # pylint: disable=no-self-use # not yet!
    def apply_target_transforms(
        self, target: torch.Tensor, dtype: torch.dtype = torch.long
    ) -> torch.Tensor:
        """Apply transforms to the target.
        Note:
            This is useful for tasks such as segmentation object detection where
            targets are in the form of bounding boxes, segmentation masks etc.
        """
        return torch.tensor(target, dtype=dtype)


class TFImageClassificationDataset(tf.keras.utils.Sequence):
    """Template for Image Classification Dataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        stage: str = "train",
        batch_size=1,
        num_classes=2,
        target_size=(32, 32),
        shuffle=False,
        transforms=None,
        num_channels=3,
        x_col="",
        y_col="",
        **kwargs,
    ) -> None:
        """"""

        self.df: pd.DataFrame = df
        self.stage: str = stage
        self.transforms = transforms
        self.batch_size: int = batch_size
        self.num_classes: int = num_classes
        self.dim: list | tuple = target_size
        self.num_channels: int = num_channels
        self.shuffle: bool = shuffle

        self.list_IDs = df[x_col].values
        self.targets = df[y_col].values if stage != "test" else None
        self._on_epoch_end()

    def __len__(self) -> int:
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index: int):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp: list[Any | None] = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self._data_generation(list_IDs_temp, indexes)

        # TODO: consider stage to be private since it is only used internally.
        if self.stage in ["train", "valid", "debug"]:
            return X, y
        elif self.stage == "test":
            return X
        else:
            raise ValueError(f"Invalid stage {self.stage}.")

    def load_image(self, image_path):
        image: np.ndarray[int, np.dtype[np.generic]] = cv2.imread(image_path)  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.apply_image_transforms(image)
        """ Preprocessed numpy.array or a tf.Tensor with type float32. The images are converted from RGB to BGR,
            then each color channel is zero-centered with respect to the ImageNet dataset, without scaling. """
        return image

    def _data_generation(self, list_IDs_temp, indexes):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.num_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i] = self.load_image(ID)

        # Store class
        y = self.targets[indexes]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.num_classes)

    def apply_image_transforms(self, image):
        """Apply transforms to the image."""
        if self.transforms and isinstance(self.transforms, A.Compose):
            image = self.transforms(image=image)["image"]
        elif self.transforms and isinstance(self.transforms, T.Compose):
            image = self.transforms(image)
        return image

    def _on_epoch_end(self) -> None:
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
