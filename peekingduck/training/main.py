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

"""Controller for training pipeline."""

import logging

import hydra
from configs.base import Config
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

# from hydra.utils import instantiate

logger: logging.Logger = logging.getLogger(__name__)


def hydra_to_pydantic(config: DictConfig) -> Config:
    """Converts Hydra config to Pydantic config."""
    OmegaConf.resolve(config)
    return Config(**config)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main entry to training pipeline."""
    logger.info(f"Config representation:\n{OmegaConf.to_yaml(config)}")
    logger.info(f"Output dir: {HydraConfig.get().runtime.output_dir}")

    # callbacks = instantiate(config.callbacks.callbacks)

    config = hydra_to_pydantic(config)
    pprint(config)

    # pass config to training pipeline
    # run(config)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
