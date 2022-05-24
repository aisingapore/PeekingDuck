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

import hashlib
import operator
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import requests
from tqdm import tqdm

BASE_URL = "https://storage.googleapis.com/peekingduck/models"
PEEKINGDUCK_WEIGHTS_SUBDIR = "peekingduck_weights"

Number = Union[float, int]


class ThresholdCheckerMixin:
    """Mixin class providing utility methods for checking validity of config
    values, typically thresholds.
    """

    def check_bounds(
        self,
        key: Union[str, List[str]],
        value: Union[Number, Tuple[Number, Number]],
        method: str,
        include: Optional[str] = "both",
    ) -> None:
        """Checks if the configuration value(s) specified by `key` satisties
        the specified bounds.

        Args:
            key (Union[str, List[str]]): The specified key or list of keys.
            value (Union[Number, Tuple[Number, Number]]): Either a single
                number to specify the upper or lower bound or a tuple of
                numbers to specify both the upper and lower bounds.
            method (str): The bounds checking methods, one of
                {"above", "below", "both"}. If "above", checks if the
                configuration value is above the specified `value`. If "below",
                checks if the configuration value is below the specified
                `value`. If "both", checks if the configuration value is above
                `value[0]` and below `value[1]`.
            include (Optional[str]): Indicates if the `value` itself should be
                included in the bound, one of {"lower", "upper", "both", None}.
                Please see Technotes for details.

        Raises:
            TypeError: `key` type is not in (List[str], str).
            TypeError: If `value` is not a tuple of only float/int.
            TypeError: If `value` is not a tuple with 2 elements.
            TypeError: If `value` is not a float, int, or tuple.
            TypeError: If `value` type is not a tuple when `method` is
                "within".
            TypeError: If `value` type is a tuple when `method` is
                "above"/"below".
            ValueError: If `method` is not one of {"above", "below", "within"}.
            ValueError: If the configuration value fails the bounds comparison.

        Technotes:
            The behavior of `include` depends on the specified `method`. The
            table below shows the comparison done for various argument
            combinations.

            +-----------+---------+-------------------------------------+
            | method    | include | comparison                          |
            +===========+=========+=====================================+
            |           | "lower" | config[key] >= value                |
            +           +---------+-------------------------------------+
            |           | "upper" | config[key] > value                 |
            +           +---------+-------------------------------------+
            |           | "both"  | config[key] >= value                |
            +           +---------+-------------------------------------+
            | "above"   | None    | config[key] > value                 |
            +-----------+---------+-------------------------------------+
            |           | "lower" | config[key] < value                 |
            +           +---------+-------------------------------------+
            |           | "upper" | config[key] <= value                |
            +           +---------+-------------------------------------+
            |           | "both"  | config[key] <= value                |
            +           +---------+-------------------------------------+
            | "below"   | None    | config[key] < value                 |
            +-----------+---------+-------------------------------------+
            |           | "lower" | value[0] <= config[key] < value[1]  |
            +           +---------+-------------------------------------+
            |           | "upper" | value[0] < config[key] <= value[1]  |
            +           +---------+-------------------------------------+
            |           | "both"  | value[0] <= config[key] <= value[1] |
            +           +---------+-------------------------------------+
            | "within"  | None    | value[0] < config[key] < value[1]   |
            +-----------+---------+-------------------------------------+
        """
        # available checking methods
        methods = {"above", "below", "within"}
        # available options of lower/upper bound inclusion
        lower_includes = {"lower", "both"}
        upper_includes = {"upper", "both"}

        if method not in methods:
            raise ValueError(f"`method` must be one of {methods}")

        if isinstance(value, tuple):
            if not all(isinstance(val, (float, int)) for val in value):
                raise TypeError(
                    "When using tuple for `value`, it must be a tuple of float/int"
                )
            if len(value) != 2:
                raise ValueError(
                    "When using tuple for `value`, it must contain only 2 elements"
                )
        elif isinstance(value, (float, int)):
            pass
        else:
            raise TypeError(
                "`value` must be a float/int or tuple, but you passed a "
                f"{type(value).__name__}"
            )

        if method == "within":
            if not isinstance(value, tuple):
                raise TypeError("`value` must be a tuple when `method` is 'within'")
            self._check_within_bounds(
                key, value, (include in lower_includes, include in upper_includes)
            )
        else:
            if isinstance(value, tuple):
                raise TypeError(
                    "`value` must be a float/int when `method` is 'above'/'below'"
                )
            if method == "above":
                self._check_above_value(key, value, include in lower_includes)
            elif method == "below":
                self._check_below_value(key, value, include in upper_includes)

    def check_valid_choice(
        self, key: str, choices: Set[Union[int, float, str]]
    ) -> None:
        """Checks that configuration value specified by `key` can be found
        in `choices`.

        Args:
            key (str): The specified key.
            choices (Set[Union[int, float, str]]): The valid choices.

        Raises:
            TypeError: `key` type is not a str.
            ValueError: If the configuration value is not found in `choices`.
        """
        if not isinstance(key, str):
            raise TypeError("`key` must be str")
        if self.config[key] not in choices:
            raise ValueError(f"{key} must be one of {choices}")

    def _check_above_value(
        self, key: Union[str, List[str]], value: Number, inclusive: bool
    ) -> None:
        """Checks that configuration values specified by `key` is more than
        (or equal to) the specified `value`.

        Args:
            key (Union[str, List[str]]): The specified key or list of keys.
            value (Number): The specified value.
            inclusive (bool): If `True`, compares `config[key] >= value`. If
                `False`, compares `config[key] > value`.

        Raises:
            TypeError: `key` type is not in (List[str], str).
            ValueError: If the configuration value is less than (or equal to)
                `value`.
        """
        method = operator.ge if inclusive else operator.gt
        extra_reason = " or equal to" if inclusive else ""
        self._compare(key, value, method, reason=f"more than{extra_reason} {value}")

    def _check_below_value(
        self, key: Union[str, List[str]], value: Number, inclusive: bool
    ) -> None:
        """Checks that configuration values specified by `key` is more than
        (or equal to) the specified `value`.

        Args:
            key (Union[str, List[str]]): The specified key or list of keys.
            value (Number): The specified value.
            inclusive (bool): If `True`, compares `config[key] <= value`. If
                `False`, compares `config[key] < value`.

        Raises:
            TypeError: `key` type is not in (List[str], str).
            ValueError: If the configuration value is less than (or equal to)
                `value`.
        """
        method = operator.le if inclusive else operator.lt
        extra_reason = " or equal to" if inclusive else ""
        self._compare(key, value, method, reason=f"less than{extra_reason} {value}")

    def _check_within_bounds(
        self,
        key: Union[str, List[str]],
        bounds: Tuple[Number, Number],
        includes: Tuple[bool, bool],
    ) -> None:
        """Checks that configuration values specified by `key` is within the
        specified bounds between `lower` and `upper`.

        Args:
            key (Union[str, List[str]]): The specified key or list of keys.
             (Union[float, int]): The lower bound.
            bounds (Tuple[Number, Number]): The lower and upper bounds.
            includes (Tuple[bool, bool]): If `True`, compares `config[key] >= value`.
                If `False`, compares `config[key] > value`.
            inclusive_upper (bool): If `True`, compares `config[key] <= value`.
                If `False`, compares `config[key] < value`.

        Raises:
            TypeError: `key` type is not in (List[str], str).
            ValueError: If the configuration value is not between `lower` and
                `upper`.
        """
        method_lower = operator.ge if includes[0] else operator.gt
        method_upper = operator.le if includes[1] else operator.lt
        reason_lower = "[" if includes[0] else "("
        reason_upper = "]" if includes[1] else ")"
        reason = f"between {reason_lower}{bounds[0]}, {bounds[1]}{reason_upper}"
        self._compare(key, bounds[0], method_lower, reason)
        self._compare(key, bounds[1], method_upper, reason)

    def _compare(
        self,
        key: Union[str, List[str]],
        value: Union[float, int],
        method: Callable,
        reason: str,
    ) -> None:
        """Compares the configuration values specified by `key` with
        `value` using the specified comparison `method`, raises error with
        `reason` if comparison fails.

        Args:
            key (Union[str, List[str]]): The specified key or list of keys.
            value (Union[float, int]): The specified value.
            method (Callable): The method to be used to compare the
                configuration value specified by `key` and `value`.
            reason (str): The failure reason.

        Raises:
            TypeError: `key` type is not in (List[str], str).
            ValueError: If the comparison between `config[key]` and `value`
                fails.
        """
        if isinstance(key, str):
            if isinstance(self.config[key], list):
                if not all(method(val, value) for val in self.config[key]):
                    raise ValueError(f"All elements of {key} must be {reason}")
            elif not method(self.config[key], value):
                raise ValueError(f"{key} must be {reason}")
        elif isinstance(key, list):
            for k in key:
                self._compare(k, value, method, reason)
        else:
            raise TypeError("`key` must be either str or list")


class WeightsDownloaderMixin:
    """Mixin class providing utility methods for downloading model weights."""

    @property
    def blob_filename(self) -> str:
        """Name of the selected weights on GCP."""
        return self.weights["blob_file"][self.config["model_type"]]

    @property
    def classes_filename(self) -> Optional[str]:
        """Name of the file containing classes IDs/labels for the selected
        model.
        """
        return self.weights.get("classes_file")

    @property
    def weights(self) -> Dict[str, Any]:
        """Dictionary of `blob_file`, `config_file`, and `model_file` names
        based on the selected `model_format`.
        """
        return self.config["weights"][self.config["model_format"]]

    @property
    def model_filename(self) -> str:
        """Name of the selected weights on local machine."""
        return self.weights["model_file"][self.config["model_type"]]

    @property
    def model_subdir(self) -> str:
        """Model weights sub-directory name based on the selected model
        format.
        """
        return self.weights["model_subdir"]

    def download_weights(self) -> Path:
        """Downloads weights for specified ``blob_file``.

        Returns:
            (Path): Path to the directory where the model's weights are stored.
        """
        model_dir = self._find_paths()
        if self._has_weights(model_dir):
            return model_dir

        self.logger.info("Proceeding to download...")

        model_dir.mkdir(parents=True, exist_ok=True)
        self._download_to(self.blob_filename, model_dir)
        self._extract_file(model_dir)
        if self.classes_filename is not None:
            self._download_to(self.classes_filename, model_dir)

        self.logger.info(f"Weights downloaded to {model_dir}.")

        return model_dir

    def _download_to(self, filename: str, destination_dir: Path) -> None:
        """Downloads publicly shared files from Google Cloud Platform.

        Saves download content in chunks. Chunk size set to large integer as
        weights are usually pretty large.

        Args:
            destination_dir (Path): Destination directory of downloaded file.
        """
        with open(destination_dir / filename, "wb") as outfile, requests.get(
            f"{BASE_URL}/{self.model_subdir}/{self.config['model_format']}/{filename}",
            stream=True,
        ) as response:
            for chunk in tqdm(response.iter_content(chunk_size=32768)):
                if chunk:  # filter out keep-alive new chunks
                    outfile.write(chunk)

    def _extract_file(self, destination_dir: Path) -> None:
        """Extracts the zip file to ``destination_dir``.

        Args:
            destination_dir (Path): Destination directory for extraction.
        """
        zip_path = destination_dir / self.blob_filename
        with zipfile.ZipFile(zip_path, "r") as infile:
            file_list = infile.namelist()
            for file in tqdm(file=sys.stdout, iterable=file_list, total=len(file_list)):
                infile.extract(member=file, path=destination_dir)

        os.remove(zip_path)

    def _find_paths(self) -> Path:
        """Constructs the `peekingduck_weights` directory path and the model
        sub-directory path.

        Returns:
            (Path): /path/to/peekingduck_weights/<model_name> where
              weights for a model are stored.

        Raises:
            FileNotFoundError: When the user-specified `weights_parent_dir`
                does not exist.
            ValueError: When the user-specified `weights_parent_dir` is not an
                absolute path.
        """
        if self.config["weights_parent_dir"] is None:
            weights_parent_dir = self.config["root"].parent
        else:
            weights_parent_dir = Path(self.config["weights_parent_dir"])

            if not weights_parent_dir.exists():
                raise FileNotFoundError(
                    f"weights_parent_dir does not exist: {weights_parent_dir}"
                )
            if not weights_parent_dir.is_absolute():
                raise ValueError(
                    f"weights_parent_dir must be an absolute path: {weights_parent_dir}"
                )

        return (
            weights_parent_dir
            / PEEKINGDUCK_WEIGHTS_SUBDIR
            / self.model_subdir
            / self.config["model_format"]
        )

    def _get_weights_checksum(self) -> str:
        with requests.get(f"{BASE_URL}/weights_checksums.json") as response:
            checksums = response.json()
        return checksums[self.model_subdir][self.config["model_format"]][
            str(self.config["model_type"])
        ]

    def _has_weights(self, model_dir: Path) -> bool:
        """Checks if the specified weights file is present in the model
        sub-directory of the PeekingDuck weights directory.

        Args:
            model_dir (Path): /path/to/peekingduck_weights/<model_name> where
                weights for a model are stored.

        Returns:
            (bool): ``True`` if specified weights file in ``model_dir``
            exists and up-to-date/not corrupted, else ``False``.
        """
        weights_path = model_dir / self.model_filename
        # self.logger.info(f"has_weights: weights_path={weights_path}")

        if not weights_path.exists():
            self.logger.warning("No weights detected.")
            return False
        # dotw: temp code for onnx/tensorrt to guarantee success
        if str(weights_path).endswith(("onnx", "trt", "fp16")):
            # self.logger.info("temp code override: return True")
            return True
        if self.sha256sum(weights_path).hexdigest() != self._get_weights_checksum():
            self.logger.warning("Weights file is corrupted/out-of-date.")
            return False
        return True

    @staticmethod
    def sha256sum(path: Path, hash_func: "hashlib._Hash" = None) -> "hashlib._Hash":
        """Hashes the specified file/directory using SHA256. Reads the file in
        chunks to be more memory efficient.

        When a directory path is passed as the argument, sort the folder
        content and hash the content recursively.

        Args:
            path (Path): Path to the file to be hashed.
            hash_func (Optional[hashlib._Hash]): A hash function which uses the
                SHA-256 algorithm.

        Returns:
            (hashlib._Hash): The updated hash function.
        """
        if hash_func is None:
            hash_func = hashlib.sha256()

        if path.is_dir():
            for subpath in sorted(path.iterdir()):
                if subpath.name not in {".DS_Store", "__MACOSX"}:
                    hash_func = WeightsDownloaderMixin.sha256sum(subpath, hash_func)
        else:
            buffer_size = hash_func.block_size * 1024
            with open(path, "rb") as infile:
                for chunk in iter(lambda: infile.read(buffer_size), b""):
                    hash_func.update(chunk)
        return hash_func
