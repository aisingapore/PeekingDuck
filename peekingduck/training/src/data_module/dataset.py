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
from pathlib import Path
from typing import Optional, Tuple, Union
from omegaconf import DictConfig

import albumentations as A
import cv2
import pandas as pd
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

TransformTypes = Optional[Union[A.Compose, T.Compose]]


class ImageClassificationDataset(Dataset):
    """A sample template for Image Classification Dataset."""

    def __init__(
        self,
        cfg: DictConfig,
        df: Optional[pd.DataFrame] = None,
        stage: str = "train",
        **kwargs,
    ) -> None:
        """"""

        super().__init__(**kwargs)
        self.cfg: DictConfig = cfg
        self.df = df
        self.stage = stage

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.df)

    def __getitem__(
        self, index: int
    ) -> Union[torch.FloatTensor, Union[torch.FloatTensor, torch.LongTensor]]:
        """ """
        image_path = self.image_path[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.apply_image_transforms(image)

        # Get target for all modes except for test dataset.
        # If test, replace target with dummy ones as placeholder.
        target = self.targets[index] if self.stage != "test" else torch.ones(1)
        target = self.apply_target_transforms(target)

        # TODO: consider stage to be private since it is only used internally.
        if self.stage in ["train", "valid", "debug"]:
            return image, target
        elif self.stage == "test":
            return image
        elif self.stage == "gradcam":
            # get image id as well to show on matplotlib image!
            return image, target, self.image_ids[index]
        else:
            raise ValueError(f"Invalid stage {self.stage}.")

    def apply_image_transforms(
        self, image: torch.Tensor, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Apply transforms to the image."""
        if self.transforms and isinstance(self.transforms, A.Compose):
            image = self.transforms(image=image)["image"]
        elif self.transforms and isinstance(self.transforms, T.Compose):
            image = self.transforms(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)  # convert HWC to CHW
        return torch.tensor(image, dtype=dtype)

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

    # @classmethod
    # def from_df(
    #     self,
    #     cfg: DictConfig,
    # ):
    #     """Creates an instance of the dataset class from a dataframe."""
