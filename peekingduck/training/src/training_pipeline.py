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

from omegaconf import DictConfig
from time import perf_counter
import logging
from hydra.utils import instantiate
from configs import LOGGER_NAME

from src.data.base import DataModule
from src.trainer.base import Trainer
from src.utils.general_utils import choose_torch_device, set_tensorflow_device
from src.model_analysis.weights_biases import WeightsAndBiases

logger: logging.Logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name


def run(cfg: DictConfig) -> None:
    assert cfg.framework in [
        "pytorch",
        "tensorflow",
    ], f"Unsupported framework {cfg.framework}"
    start_time: float = perf_counter()

    if cfg.framework == "pytorch":
        cfg.device = choose_torch_device()
        logger.info(f"Using device: {cfg.device}")
    if cfg.framework == "tensorflow":
        logger.info(set_tensorflow_device())

    data_module: DataModule = instantiate(
        config=cfg.data_module.module,
        cfg=cfg.data_module,
    )
    data_module.prepare_data()
    data_module.setup(stage="fit")
    train_loader = data_module.get_train_dataset()
    validation_loader = data_module.get_validation_dataset()

    model_analysis: WeightsAndBiases = WeightsAndBiases(cfg.model_analysis)

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

    end_time: float = perf_counter()
    run_time: str = f"Run time = {end_time - start_time:.2f} sec"
    logger.info(run_time)
    model_analysis.log({"run_time": end_time - start_time})
