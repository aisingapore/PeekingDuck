# Copyright 2022 AI Singapore
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

"""Mixin classes for PeekingDuck nodes and models."""

import os
import zipfile
from pathlib import Path
from typing import List, Tuple, Union

import requests
from tqdm import tqdm

BASE_URL = "https://storage.googleapis.com/peekingduck/models"
PEEKINGDUCK_WEIGHTS_SUBDIR = "peekingduck_weights"


class ThresholdCheckerMixin:
    """Mixin class providing utility methods for checking validity of config
    values, typically thresholds.
    """

    def ensure_above_value(self, key: Union[str, List[str]], value: float) -> None:
        """Checks that configuration values specified by ``key`` is more than
        the specified ``value``.

        Args:
            key (Union[str, List[str]]): The specified key or list of keys.
            value (float): The specified value.

        Raises:
            TypeError: ``key`` is not in (List[str], str).
            ValueError: If the configuration value is <=value.
        """
        if isinstance(key, str):
            if self.config[key] <= value:
                raise ValueError(f"{key} must be more than {value}")
        elif isinstance(key, list):
            for k in key:
                self.ensure_above_value(k, value)
        else:
            raise TypeError("'key' must be either 'str' or 'list'")

    def ensure_within_bounds(
        self, key: Union[str, List[str]], lower: float, upper: float
    ) -> None:
        """Checks that configuration values specified by ``key`` is within the
        specified bounds [start, stop].

        Args:
            key (Union[str, List[str]]): The specified key or list of keys.
            lower (float): The lower bound.
            upper (float): The upper bound.

        Raises:
            TypeError: ``key`` is not in (List[str], str).
            ValueError: If the configuration value is <=value.
        """
        if isinstance(key, str):
            if not lower <= self.config[key] <= upper:
                raise ValueError(f"{key} must be between [{lower}, {upper}]")
        elif isinstance(key, list):
            for k in key:
                self.ensure_within_bounds(k, lower, upper)
        else:
            raise TypeError("'key' must be either 'str' or 'list'")


class WeightsDownloaderMixin:
    """Mixin class providing utility methods for downloading model weights."""

    def download_weights(self) -> Path:
        """Downloads weights for specified ``blob_file``.

        Returns:
            (Path): Path to the directory where the model's weights are stored.
        """
        weights_dir, model_dir = self._find_paths()
        if self._has_weights(weights_dir, model_dir):
            return model_dir

        self.logger.warning("No weights detected. Proceeding to download...")

        zip_path = weights_dir / "temp.zip"
        self._download_blob_to(zip_path)
        self.extract_file(zip_path, weights_dir)

        self.logger.info(f"Weights downloaded to {weights_dir}.")

        return model_dir

    def _download_blob_to(self, destination: Path) -> None:
        """Downloads publicly shared files from Google Cloud Platform.

        Saves download content in chunks. Chunk size set to large integer as
        weights are usually pretty large.

        Args:
            destination (Path): Destination path of download.
        """
        with open(destination, "wb") as outfile, requests.get(
            f"{BASE_URL}/{self.config['weights']['blob_file']}", stream=True
        ) as response:
            for chunk in tqdm(response.iter_content(chunk_size=32768)):
                if chunk:  # filter out keep-alive new chunks
                    outfile.write(chunk)

    def _find_paths(self) -> Tuple[Path, Path]:
        """Checks for model weight paths from weights folder.

        Returns:
            weights_dir (Path): Path to where all weights are stored.
            model_dir (Path): Path to where weights for a model are stored.
        """
        if self.config["weights_parent_dir"] is None:
            weights_dir = self.config["root"].parent / PEEKINGDUCK_WEIGHTS_SUBDIR
        else:
            weights_parent_dir = Path(self.config["weights_parent_dir"])
            if not weights_parent_dir.exists():
                raise FileNotFoundError(
                    f"The specified weights_parent_dir: {weights_parent_dir} does not exist."
                )
            if not weights_parent_dir.is_absolute():
                raise ValueError(
                    f"The specified weights_parent_dir: {weights_parent_dir} "
                    "must be an absolute path."
                )
            weights_dir = weights_parent_dir / PEEKINGDUCK_WEIGHTS_SUBDIR

        model_dir = weights_dir / self.config["weights"]["model_subdir"]

        return weights_dir, model_dir

    @staticmethod
    def _has_weights(weights_dir: Path, model_dir: Path) -> bool:
        """Checks for model weight paths from weights folder.

        Args:
            weights_dir (Path): Path to where all weights are stored.
            model_dir (Path): Path to where weights for a model are stored.

        Returns:
            (bool): ``True`` if specified files/directories in
            ``weights_dir`` exist, else ``False``.
        """
        if not weights_dir.exists():
            weights_dir.mkdir()
            return False
        # Doesn't actually check if the files exist
        return model_dir.exists()

    @staticmethod
    def extract_file(zip_path: Path, destination_dir: Path) -> None:
        """Extracts the zip file to ``destination_dir``.

        Args:
            zip_path (Path): Path to zip file.
            destination (Path): Destination directory for extraction.
        """
        with zipfile.ZipFile(zip_path, "r") as infile:
            for file in tqdm(iterable=infile.namelist(), total=len(infile.namelist())):
                infile.extract(member=file, path=destination_dir)

        os.remove(zip_path)
