#!/usr/bin/env python
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

"""Main function for training pipeline"""

import logging

from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig

from src.training_pipeline import run

logger: logging.Logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """main"""
    logger.debug(OmegaConf.to_yaml(cfg))
    logger.info(f"runtime.output_dir{HydraConfig.get().runtime.output_dir}")
    run(cfg)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    main()
