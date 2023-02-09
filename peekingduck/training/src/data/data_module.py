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
from typing import Dict, List, Optional, Tuple

from omegaconf import DictConfig
from hydra.utils import instantiate
import pandas as pd

from src.data.base import DataModule
from src.data.dataset import PTImageClassificationDataset
from src.data.data_adapter import DataAdapter
from src.transforms.augmentations import ImageClassificationTransforms
from src.utils.general_utils import (
    create_dataframe_with_image_info,
    download_to,
    extract_file,
    return_list_of_files,
    stratified_sample_df,
)

import logging
from configs import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name


class ImageClassificationDataModule(DataModule):
    """Data module for generic image classification dataset."""

    def __init__(
        self,
        cfg: DictConfig,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cfg: DictConfig = cfg
        self.transforms: ImageClassificationTransforms = ImageClassificationTransforms(
            cfg.transform[cfg.framework]
        )
        self.dataset_loader: DataAdapter = None  # Setup in self.setup()

    def get_train_dataset(self):
        """ """
        assert self.dataset_loader is not None, "call setup() before getting dataloader"
        return self.dataset_loader.train_dataloader(
            self.train_dataset,
            transforms=self.train_transforms,
        )

    def get_validation_dataset(self):
        """ """
        assert self.dataset_loader is not None, "call setup() before getting dataloader"
        return self.dataset_loader.valid_dataloader(
            self.valid_dataset,
            transforms=self.valid_transforms,
        )

    def get_test_dataloader(self):
        """ """
        assert self.dataset_loader is not None, "call setup() before getting dataloader"
        return self.dataset_loader.test_dataloader(
            self.test_dataset,
            transforms=self.test_transforms,
        )

    def prepare_data(self) -> None:
        """Step 2 after __init__()
        Load the dataframe from user-defined csv file"""
        url: str = self.cfg.dataset.url
        blob_file: str = self.cfg.dataset.blob_file
        root_dir: Path = Path(self.cfg.dataset.root_dir)
        train_dir: Path = Path(self.cfg.dataset.train_dir)
        test_dir: Path = Path(self.cfg.dataset.test_dir)
        class_name_to_id: Dict[str, int] = self.cfg.dataset.class_name_to_id
        train_csv: str = self.cfg.dataset.train_csv
        stratify_by = self.cfg.dataset.stratify_by

        if self.cfg.dataset.download:
            logger.info(f"downloading from {url} to {blob_file} in {root_dir}")
            download_to(url, blob_file, root_dir)
            extract_file(root_dir, blob_file)

        train_images: List[str] | List[Path] = return_list_of_files(
            train_dir, extensions=[".jpg", ".png", ".jpeg"], return_string=False
        )
        test_images: List[str] | List[Path] = return_list_of_files(
            test_dir, extensions=[".jpg", ".png", ".jpeg"], return_string=False
        )
        logger.info(f"Total number of images: {len(train_images)}")
        logger.info(f"Total number of test images: {len(test_images)}")

        if Path(train_csv).exists():
            # this step is assumed to be done by user where
            # image_path is inside the csv.
            df: pd.DataFrame = pd.read_csv(train_csv)
        else:
            # TODO: only invoke this if images are store in the following format
            # train_dir
            #   - class1 ...
            df: pd.DataFrame = create_dataframe_with_image_info(
                train_images,
                class_name_to_id,
                save_path=train_csv,
            )
        logger.info(df.head())
        self.train_df = df
        self.train_df, self.test_df = self._cross_validation_split(
            self.cfg.resample.resample_strategy, df, stratify_by=stratify_by
        )
        self.train_df, self.valid_df = self._cross_validation_split(
            self.cfg.resample.resample_strategy, self.train_df, stratify_by=stratify_by
        )

        if self.cfg.debug:
            num_debug_samples = self.cfg.num_debug_samples
            logger.info(
                f"Debug mode is on, using {num_debug_samples} images for training."
            )
            if stratify_by is None:
                self.train_df = self.train_df.sample(num_debug_samples)
                self.valid_df = self.valid_df.sample(num_debug_samples)
            else:
                self.train_df = stratified_sample_df(
                    self.train_df, stratify_by, num_debug_samples
                )
                self.valid_df = stratified_sample_df(
                    self.valid_df, stratify_by, num_debug_samples
                )

        logger.info(self.train_df.info())
        # self.train_df.to_csv("train_df.csv")
        # self.valid_df.to_csv("valid_df.csv")

    def setup(self, stage: str):
        """Step 3 after prepare()"""
        self.train_transforms = self.transforms.train_transforms
        self.valid_transforms = self.transforms.valid_transforms
        self.test_transforms = self.transforms.test_transforms
        if stage == "fit":
            if self.cfg.framework == "pytorch":
                self.train_dataset = PTImageClassificationDataset(
                    self.cfg,
                    df=self.train_df,
                    stage="train",
                    transforms=self.train_transforms,
                )
                self.valid_dataset = PTImageClassificationDataset(
                    self.cfg,
                    df=self.valid_df,
                    stage="valid",
                    transforms=self.valid_transforms,
                )
                self.test_dataset = PTImageClassificationDataset(
                    self.cfg,
                    df=self.test_df,
                    stage="test",
                    transforms=self.test_transforms,
                )
            if self.cfg.framework == "tensorflow":
                self.train_dataset = self.train_df
                self.valid_dataset = self.valid_df
                self.test_dataset = self.test_df

        self.dataset_loader: DataAdapter = DataAdapter(
            self.cfg.data_adapter[self.cfg.framework]
        )

    @staticmethod
    def _cross_validation_split(
        resample_strategy: DictConfig,
        df: pd.DataFrame,
        fold: Optional[int] = None,
        stratify_by: Optional[list] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the dataframe into train and validation dataframes."""
        resample_func = instantiate(resample_strategy)

        if stratify_by is None:
            return resample_func(df)

        logger.info(f"stratify_by: {stratify_by}")
        return resample_func(df, stratify=df[stratify_by])
