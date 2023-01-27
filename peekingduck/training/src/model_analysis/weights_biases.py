import wandb
from omegaconf import DictConfig
import pandas as pd


class WeightsAndBiases:
    def __init__(self, cfg: DictConfig):
        self.cfg: DictConfig = cfg
        wandb.init(project=cfg.project, entity="peekingduck")
        wandb.config = cfg

    def watch(self, model):
        wandb.watch(model)

    def log(self, loss):
        wandb.log(loss)

    def log_history(self, history):
        selected_history = {
            key: history[key]
            for key in [
                # "train_loss",
                "valid_loss",
                "valid_elapsed_time",
                "val_MulticlassAccuracy",
                "val_MulticlassPrecision",
                "val_MulticlassRecall",
                "val_MulticlassAUROC",
            ]
        }

        df: pd.DataFrame = pd.DataFrame(selected_history)
        for row_dict in df.to_dict(orient="records"):
            wandb.log(row_dict)

    def log_training_loss(self, loss):
        wandb.log({"train_loss": loss})

    def log_validation_loss(self, loss):
        wandb.log({"val_loss": loss})
