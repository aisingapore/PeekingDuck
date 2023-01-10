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
from typing import Dict

from omegaconf import DictConfig
import pandas as pd

from src.data_module.base import DataModule
from src.utils.general_utils import (
    create_dataframe_with_image_info,
    download_to,
    extract_file,
    return_filepath,
    return_list_of_files,
    show,
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

    def setup(self):
        """ """

    def train_dataloader(self):
        """ """

    def valid_dataloader(self):
        """ """

    def prepare_data(self):
        """ """
        url = self.cfg.data_set.url
        blob_file = self.cfg.data_set.blob_file
        root_dir: Path = Path(self.cfg.data_set.root_dir)
        train_dir: Path = Path(self.cfg.data_set.train_dir)
        test_dir: Path = Path(self.cfg.data_set.test_dir)
        class_name_to_id: Dict[str, int] = self.cfg.data_set.class_name_to_id
        train_csv = self.cfg.data_set.train_csv

        if self.cfg.data_set.download:
            logger.info(f"downloading from {url} to {blob_file} in {root_dir}")
            download_to(url, blob_file, root_dir)
            extract_file(root_dir, blob_file)

        train_images = return_list_of_files(
            train_dir, extensions=[".jpg", ".png", ".jpeg"], return_string=False
        )
        test_images = return_list_of_files(
            test_dir, extensions=[".jpg", ".png", ".jpeg"], return_string=False
        )
        logger.info(f"Total number of images: {len(train_images)}")
        logger.info(f"Total number of test images: {len(test_images)}")

        if Path(train_csv).exists():
            # TODO: this step is assumed to be done by user where
            # image_path is inside the csv.
            df = pd.read_csv(train_csv)
        else:
            # TODO: only invoke this if images are store in the following format
            # train_dir
            #   - class1 ...
            df = create_dataframe_with_image_info(
                train_images,
                class_name_to_id,
                save_path=train_csv,
            )
        logger.info(df.head())
