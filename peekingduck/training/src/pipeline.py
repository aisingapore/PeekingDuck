from omegaconf import DictConfig
from time import perf_counter
import logging
from configs import LOGGER_NAME

from src.data_module.base import DataLoader
from src.model.default_model import Model
from src.trainer.default_trainer import Trainer

logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name

def run(cfg: DictConfig) -> None:
    
    start_time = perf_counter()

    dl = DataLoader()
    train_loader = dl.train_dataloader()
    valid_loader = dl.valid_dataloader()

    # TODO Support Early Stopping and Checkpoint
    callbacks = cfg.callbacks

    model = Model(cfg.model)
    trainer = Trainer(cfg.trainer, model=model, callbacks=callbacks)
    history = trainer.fit(train_loader, valid_loader)

    end_time = perf_counter()
    logger.debug(f"Run time = {end_time - start_time:.2f} sec")
