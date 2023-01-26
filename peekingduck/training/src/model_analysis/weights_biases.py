import wandb
from omegaconf import DictConfig


class WeightsAndBiases:
    def __init__(self, cfg: DictConfig):
        self.cfg: DictConfig = cfg
        wandb.init(project=cfg.project, entity="peekingduck")
        wandb.config = cfg

    def watch(model):
        wandb.watch(model)

    def log_training_loss(loss):
        wandb.log({"train_loss": loss})

    def log_validation_loss(loss):
        wandb.log({"val_loss": loss})
