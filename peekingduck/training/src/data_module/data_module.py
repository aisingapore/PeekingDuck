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

from torch.utils.data import DataLoader

from src.data_module.base import DataModule
from src.data_module.dataset import ImageClassificationDataset
from src.utils.general_utils import (
    create_dataframe_with_image_info,
    download_to,
    extract_file,
    return_list_of_files,
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
        """ """

        super().__init__(**kwargs)
        self.cfg: DictConfig = cfg

    def train_dataloader(self):
        """ """
        return DataLoader(
            self.train_dataset,
            # **self.cfg.data_module.train_loader,
        )

    def valid_dataloader(self):
        """ """
        return DataLoader(
            self.valid_dataset,
            # **self.pipeline_config.datamodule.valid_loader
        )

    def prepare_data(self):
        """ """
        url: str = self.cfg.data_set.url
        blob_file: str = self.cfg.data_set.blob_file
        root_dir: Path = Path(self.cfg.data_set.root_dir)
        train_dir: Path = Path(self.cfg.data_set.train_dir)
        test_dir: Path = Path(self.cfg.data_set.test_dir)
        class_name_to_id: Dict[str, int] = self.cfg.data_set.class_name_to_id
        train_csv: str = self.cfg.data_set.train_csv

        if self.cfg.data_set.download:
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
            # TODO: this step is assumed to be done by user where
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
        self.train_df, self.valid_df = self.cross_validation_split(df)
        logger.info(self.train_df.info())

    def setup(self, stage: str):
        """ """
        # train_transforms = self.transforms.train_transforms
        # valid_transforms = self.transforms.valid_transforms
        if stage == "fit":
            self.train_dataset = ImageClassificationDataset(
                self.cfg,
                df=self.train_df,
                stage="train",
                # transforms=train_transforms,
            )
            self.valid_dataset = ImageClassificationDataset(
                self.cfg,
                df=self.valid_df,
                stage="valid",
                # transforms=valid_transforms,
            )

    def cross_validation_split(
        self, df: pd.DataFrame, fold: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the dataframe into train and validation dataframes."""
        resample_strategy = self.cfg.resample.resample_strategy
        train_df, valid_df = instantiate(resample_strategy, df)

        return train_df, valid_df
