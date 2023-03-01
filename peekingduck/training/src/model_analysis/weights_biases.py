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

from typing import Any, Dict

import wandb
from omegaconf import DictConfig
import pandas as pd


class WeightsAndBiases:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg: DictConfig = cfg
        if cfg.debug:
            wandb.init(mode="disabled")
        else:
            wandb.init(
                project=cfg.project,
                entity="peekingduck",
                config=cfg,
                name=f"{cfg.framework}",
            )

    def watch(self, model: Any) -> None:
        wandb.watch(model)

    def log(self, loss: Dict[str, Any]) -> None:
        wandb.log(loss)

    def log_history(self, history: Dict[str, Any]) -> None:
        if self.cfg.framework == "tensorflow":
            selected_history = history

            df: pd.DataFrame = pd.DataFrame(selected_history)
            for row_dict in df.to_dict(orient="records"):
                wandb.log(row_dict)

    def log_training_loss(self, loss: Dict[str, Any]) -> None:
        wandb.log({"train_loss": loss})

    def log_validation_loss(self, loss: Dict[str, Any]) -> None:
        wandb.log({"val_loss": loss})
