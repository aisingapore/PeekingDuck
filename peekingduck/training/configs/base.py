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

"""Base class for config abstractions."""

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, conint, validator  # pylint:disable=no-name-in-module
from sklearn import model_selection

# from src.callbacks.history import History

# pylint: disable=no-self-argument, no-self-use, too-few-public-methods
class Dataset(BaseModel):
    """Base config for dataset."""

    # filepath configs
    root_dir: Path
    train_dir: Path
    train_csv: Path
    test_dir: Path
    test_csv: Path

    # dataset configs
    image_col_name: str
    image_path_col_name: str
    target_col_name: str
    class_name_to_id: Dict[str, int]
    group_by: Optional[str] = None
    stratify_by: Optional[str] = None
    url: Optional[str] = None
    blob_file: Optional[str] = None


class Resample(BaseModel):
    """Base config for resampling strategy."""

    resample_strategy: str
    resample_params: Dict[str, Any]

    @validator("resample_strategy")
    def validate_resample_strategy(cls, resample_strategy: str) -> str:
        """Validates resample_strategy is in sklearn.model_selection."""
        try:
            _ = getattr(model_selection, resample_strategy)
        except AttributeError:
            raise ValueError(
                f"resample_strategy must be in {model_selection.__all__}"
            ) from BaseException


class Model(BaseModel):
    """Base config for model."""

    model_name: str
    pretrained: bool
    in_chans: conint(ge=1)  # in_channels must be greater than or equal to 1
    num_classes: conint(ge=1)
    global_pool: str

    @validator("global_pool")
    def validate_global_pool(cls, global_pool: str) -> str:
        """Validates global_pool is in ["avg", "max"]."""
        if global_pool not in ["avg", "max"]:
            raise ValueError("global_pool must be avg or max")
        return global_pool


# class Callbacks(BaseModel):
#     """Base config for callbacks."""

#     history: History


class Stores(BaseModel):
    """Base config for stores."""

    project_name: str
    unique_id: str
    logs_dir: Path
    model_artifacts_dir: Path


class DataModule(BaseModel):
    dataset: Dataset
    resample: Resample


class General(BaseModel):
    num_classes: int
    device: str
    project_name: str
    debug: bool


class Config(BaseModel):
    datamodule: DataModule
    model: Model
    stores: Stores
    general: General
