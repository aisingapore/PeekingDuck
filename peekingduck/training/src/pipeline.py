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
from src.data_module.data_module import ImageClassificationDataModule
from src.model.model import ImageClassificationModel
from src.trainer.default_trainer import Trainer
from src.metrics.metrics_adapter import MetricsAdapter

logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name


def run(cfg: DictConfig) -> None:

    start_time = perf_counter()

    data_module: DataModule = ImageClassificationDataModule(cfg.data_module)
    data_module.prepare_data()
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    valid_loader = data_module.valid_dataloader()

    callbacks = instantiate(cfg.callbacks.callbacks)
    metricsAdapter = MetricsAdapter(
        task=cfg.classification_type, num_classes=cfg.data_module.data_set.num_classes
    )
    for metric in cfg.metrics.evaluate:
        try:
            getattr(metricsAdapter, metric)()
        except NotImplementedError:
            pass

    metrics_collection = metricsAdapter.make_collection()

    model = ImageClassificationModel(cfg.model).to(cfg.device)
    trainer = Trainer(
        cfg.trainer,
        model=model,
        callbacks=callbacks,
        metrics=metrics_collection,
        device=cfg.device,
    )
    history = trainer.fit(train_loader, valid_loader)

    end_time = perf_counter()
    logger.debug(f"Run time = {end_time - start_time:.2f} sec")
