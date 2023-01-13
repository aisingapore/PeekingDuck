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

from src.data_module.base import DataModule
from src.model.base import Model
from src.trainer.default_trainer import Trainer
from src.utils.general_utils import choose_torch_device

logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name


def run(cfg: DictConfig) -> None:

    start_time = perf_counter()

    if cfg.framework == "pytorch":
        cfg.device = choose_torch_device()  # TODO tensorflow
        logger.info(f"Using device: {cfg.device}")

    data_module: DataModule = instantiate(
        config=cfg.data_module.module,
        cfg=cfg.data_module,
    )
    data_module.prepare_data()
    data_module.setup(stage="fit")
    train_loader = data_module.get_train_dataloader()
    valid_loader = data_module.get_valid_dataloader()

    callbacks = instantiate(cfg.callbacks.callbacks)
    metrics_adapter = instantiate(cfg.metrics.adapter[cfg.framework][0])
    metrics_obj = metrics_adapter.setup(
        task=cfg.data_module.data_set.classification_type,
        num_classes=cfg.data_module.data_set.num_classes,
        metrics=cfg.metrics.evaluate,
    )

    model: Model = instantiate(
        config=cfg.model.model_type, cfg=cfg.model, _recursive_=False
    ).to(cfg.device)
    trainer = Trainer(
        cfg.trainer,
        model=model,
        callbacks=callbacks,
        metrics=metrics_obj,
        device=cfg.device,
    )
    history = trainer.fit(train_loader, valid_loader)

    end_time = perf_counter()
    logger.debug(f"Run time = {end_time - start_time:.2f} sec")
