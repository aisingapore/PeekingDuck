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


from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from omegaconf import DictConfig

import albumentations as A
import cv2
import pandas as pd

from src.config import TORCH_AVAILABLE, TF_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    from torch import Tensor
    from torch.utils.data import Dataset
    import torchvision.transforms as T
else:
    raise ImportError("Called a torch-specific function but torch is not installed.")

if TF_AVAILABLE:
    import tensorflow as tf
else:
    raise ImportError(
        "Called a tensorflow-specific function but tensorflow is not installed."
    )

TransformTypes = Optional[Union[A.Compose, T.Compose]]


class PTImageClassificationDataset(
    Dataset
):  # pylint: disable=too-many-instance-attributes, too-many-arguments
    """Template for Image Classification Dataset."""

    def __init__(
        self,
        cfg: DictConfig,
        dataframe: pd.DataFrame,
        stage: str = "train",
        transforms: TransformTypes = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """"""

        super().__init__(**kwargs)
        self.cfg: DictConfig = cfg
        self.dataframe: pd.DataFrame = dataframe
        self.stage: str = stage
        self.transforms: TransformTypes = transforms

        self.image_path = dataframe[cfg.dataset.image_path_col_name].values
        self.targets = (
            dataframe[cfg.dataset.target_col_id].values if stage != "test" else None
        )

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataframe.index)

    def __getitem__(self, index: int) -> Union[Tuple, Any]:
        """Generate one batch of data"""
        assert self.stage in [
            "train",
            "valid",
            "debug",
            "test",
        ], f"Invalid stage {self.stage}."

        image_path: str = self.image_path[index]
        image: Tensor = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.apply_image_transforms(image)

        # Get target for all modes except for test dataset.
        # If test, replace target with dummy ones as placeholder.
        target = self.targets[index] if self.stage != "test" else torch.ones(1)
        # target = self.apply_target_transforms(target)

        if self.stage in ["train", "valid", "debug"]:
            return image, target
        else:  # self.stage == "test"
            return image

    def apply_image_transforms(self, image: torch.Tensor) -> Tensor:
        """Apply transforms to the image."""
        if self.transforms and isinstance(self.transforms, A.Compose):
            image = self.transforms(image=image)["image"]
        elif self.transforms and isinstance(self.transforms, T.Compose):
            image = self.transforms(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)  # convert HWC to CHW
        return image

    def apply_target_transforms(
        self, target: torch.Tensor, dtype: torch.dtype = torch.long
    ) -> torch.Tensor:
        """Apply transforms to the target.
        Note:
            This is useful for tasks such as segmentation object detection where
            targets are in the form of bounding boxes, segmentation masks etc.
        """
        return torch.tensor(target, dtype=dtype)


class TFImageClassificationDataset(
    tf.keras.utils.Sequence
):  # pylint: disable=too-many-instance-attributes, too-many-arguments, invalid-name
    """Template for Image Classification Dataset."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        stage: str = "train",
        batch_size: int = 1,
        num_classes: int = 2,
        target_size: Union[list, tuple] = (32, 32),
        shuffle: bool = False,
        transforms: TransformTypes = None,
        num_channels: int = 3,
        x_col: str = "",
        y_col: str = "",
        **kwargs: Dict[str, Any],
    ) -> None:
        """"""

        self.dataframe = dataframe
        self.stage = stage
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.dim = target_size
        self.num_channels = num_channels
        self.shuffle = shuffle

        self.image_paths = self.dataframe[x_col].values
        self.targets = self.dataframe[y_col].values if stage != "test" else None
        self.kwargs = kwargs
        self._on_epoch_end()

    def __len__(self) -> int:
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index: int) -> Union[Tuple, Any]:
        """Generate one batch of data

        Args:
            index (int): index of the image to return.

        Returns:
            (Dict):
                Outputs dictionary with the keys `labels`.
        """
        assert self.stage in [
            "train",
            "valid",
            "debug",
            "test",
        ], f"Invalid stage {self.stage}."

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        image_paths_temp: List[Any] = [self.image_paths[k] for k in indexes]

        # Generate data
        X, y = self._data_generation(image_paths_temp, indexes)

        if self.stage in ["train", "valid", "debug"]:
            return X, y
        else:  # self.stage == "test"
            return X

    def load_image(self, image_path: str) -> Any:
        """Load image from `image_path`

        Args:
            image_path (str): image path to load.

        Returns:
            (ndarray):
                Outputs image in numpy array.
        """
        image = cv2.imread(image_path)  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.apply_image_transforms(image)
        """ Preprocessed numpy.array or a tf.Tensor with type float32. The images are converted from RGB to BGR,
            then each color channel is zero-centered with respect to the ImageNet dataset, without scaling. """
        return image

    def _data_generation(
        self,
        list_ids_temp: List[Any],
        indexes: np.ndarray,
    ) -> Any:
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty(
            (self.batch_size, *self.dim, self.num_channels)
        )  # pylint: disable=invalid-name
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, id in enumerate(list_ids_temp):
            # Store sample
            X[i] = self.load_image(id)

        # Store class
        y = self.targets[indexes]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.num_classes)

    def apply_image_transforms(self, image: Any) -> Any:
        """Apply transforms to the image."""
        if self.transforms and isinstance(self.transforms, A.Compose):
            image = self.transforms(image=image)["image"]
        elif self.transforms and isinstance(self.transforms, T.Compose):
            image = self.transforms(image)
        return image

    def _on_epoch_end(self) -> None:
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
