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
import logging

from omegaconf import OmegaConf, DictConfig
import hydra

from hydra.core.hydra_config import HydraConfig
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any

from pydantic import (
    BaseModel,
    BaseSettings,
    Field,
    PositiveInt,
    conint,
    constr,
    schema,
    schema_json_of,
    validator,
)
from pydantic.dataclasses import dataclass
from pydantic.env_settings import SettingsSourceCallable
from configs.base import Config

# from src.pipeline import run

logger: logging.Logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info(f"Project Name: {cfg.stores.project_name}")
    train_dir = cfg.datamodule.dataset.train_dir

    logger.info(f"runtime.output_dir{HydraConfig.get().runtime.output_dir}")
    OmegaConf.resolve(cfg)

    r_model = Config(**cfg)
    print(r_model)
    
    # run(cfg)


if __name__ == "__main__":
    main()
