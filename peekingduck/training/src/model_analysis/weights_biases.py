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

import wandb
from omegaconf import DictConfig
import pandas as pd


class WeightsAndBiases:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg: DictConfig = cfg
        if cfg.debug:
            wandb.init(mode="disabled")
        else:
            wandb.init(project=cfg.project, entity="peekingduck", config=cfg)

    def watch(self, model) -> None:
        wandb.watch(model)

    def log(self, loss) -> None:
        wandb.log(loss)

    def log_history(self, history) -> None:
        selected_history = {
            key: history[key]
            for key in [
                "train_loss",
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

    def log_training_loss(self, loss) -> None:
        wandb.log({"train_loss": loss})

    def log_validation_loss(self, loss) -> None:
        wandb.log({"val_loss": loss})
