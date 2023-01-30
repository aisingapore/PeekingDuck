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

from pyparsing import util
from omegaconf import DictConfig
from time import perf_counter
import logging
from hydra.utils import instantiate
from configs import LOGGER_NAME

from src.data.base import DataModule
from src.model.pytorch_base import Model
from src.trainer.base import Trainer
from src.utils.general_utils import choose_torch_device
from src.model_analysis.weights_biases import WeightsAndBiases

logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name


def run(cfg: DictConfig) -> None:

    start_time = perf_counter()

    cfg.device = choose_torch_device()  # TODO tensorflow
    logger.info(f"Using device: {cfg.device}")

    data_module: DataModule = instantiate(
        config=cfg.data_module.module,
        cfg=cfg.data_module,
    )
    data_module.prepare_data()
    data_module.setup(stage="fit")
    train_loader = data_module.get_train_dataloader()
    validation_loader = data_module.get_validation_dataloader()

    wandb = WeightsAndBiases(cfg.model_analysis)

    trainer: Trainer = instantiate(
        cfg.trainer[cfg.framework].global_train_params.trainer, cfg.framework
    )
    trainer.setup(
        cfg.trainer,
        cfg.model,
        cfg.callbacks,
        cfg.metrics,
        cfg.data_module,
        device=cfg.device,
    )
    history = trainer.train(train_loader, validation_loader)
    # wandb.log_history(history)

    end_time = perf_counter()
    logger.debug(f"Run time = {end_time - start_time:.2f} sec")
